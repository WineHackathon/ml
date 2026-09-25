from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import fetch_image, process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class QwenReranker:
    def __init__(self, model_path: Path, config_path: Path):
        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
        if model_path.name != self.config["model_id"].split("/")[-1]:
            raise ValueError("reranker model path does not match frozen config")
        if sha256(model_path / "model.safetensors") != self.config["model_safetensors_sha256"]:
            raise ValueError("reranker model weights hash mismatch")
        self.device = torch.device(os.getenv("WINE_RERANKER_DEVICE", self.config["device"]))
        if os.getenv("WINE_DISABLE_CUDNN_SDP") == "1":
            torch.backends.cuda.enable_cudnn_sdp(False)
        lm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=getattr(torch, self.config["dtype"]),
        ).to(self.device)
        self.model = lm.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, padding_side="left")
        vocab = self.processor.tokenizer.get_vocab()
        weight = lm.lm_head.weight.data[vocab["yes"]] - lm.lm_head.weight.data[vocab["no"]]
        self.score_head = torch.nn.Linear(weight.numel(), 1, bias=False, device=self.device, dtype=weight.dtype)
        with torch.no_grad():
            self.score_head.weight[0].copy_(weight)
        self.score_head.eval()
        self.profile_qwen = os.getenv("WINE_PROFILE_QWEN") == "1"
        self.preload_enabled = os.getenv("WINE_RERANKER_PRELOAD", "0") == "1"
        self.reference_images: dict[Path, Image.Image] = {}

    def _resize(self, image: Image.Image | str) -> Image.Image:
        return fetch_image({
            "image": image,
            "min_pixels": self.config["min_pixels"],
            "max_pixels": self.config["max_pixels"],
        }, image_patch_size=16)

    def preload_references(self, paths: list[Path]) -> None:
        if not self.preload_enabled:
            return
        self.reference_images = {path.resolve(): self._resize(f"file://{path.resolve()}") for path in paths}

    def close(self) -> None:
        for image in self.reference_images.values():
            image.close()
        self.reference_images.clear()

    def query_view(self, image: Image.Image) -> Image.Image:
        name = self.config["reranker_query_views"][0]
        if name == "whole":
            view = image
        elif name == "label90":
            width, height = image.size
            view = image.crop((int(width * 0.05), int(height * 0.18), int(width * 0.95), int(height * 0.94)))
        else:
            raise ValueError(f"unsupported reranker query view: {name}")
        if not self.preload_enabled:
            return view
        resized = self._resize(view)
        if view is not image:
            view.close()
        return resized

    def _messages(self, query: Image.Image, text: str, image_path: Path) -> list[dict[str, Any]]:
        reference = self.reference_images.get(image_path.resolve())
        if reference is None and self.preload_enabled:
            raise KeyError(f"reranker reference was not preloaded: {image_path}")
        reference = reference or f"file://{image_path}"
        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": 'Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".',
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f'<Instruct>: {self.config["instruction"]}'},
                    {"type": "text", "text": "<Query>:"},
                    {"type": "image", "image": query, "min_pixels": self.config["min_pixels"], "max_pixels": self.config["max_pixels"]},
                    {"type": "text", "text": "\n<Document>:"},
                    {"type": "image", "image": reference, "min_pixels": self.config["min_pixels"], "max_pixels": self.config["max_pixels"]},
                    {"type": "text", "text": text},
                ],
            },
        ]

    @torch.inference_mode()
    def score(self, query: Image.Image, text: str, image_path: Path) -> float:
        profile = self.profile_qwen
        if profile:
            timings: dict[str, float] = {}
            started = time.perf_counter()
        messages = self._messages(query, text, image_path)
        prompt = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        if profile:
            phase = time.perf_counter()
        images, videos, video_kwargs = process_vision_info(
            [messages], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
        if profile:
            timings["vision_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
        inputs = self.processor(
            text=prompt,
            images=images,
            videos=videos,
            truncation=False,
            padding=False,
            do_resize=False,
            **video_kwargs,
        )
        if len(inputs["input_ids"][0]) > self.config["max_length"]:
            raise ValueError("reranker input exceeds frozen max_length")
        tokenized = self.processor.tokenizer.pad({"input_ids": inputs["input_ids"]}, padding=True, return_tensors="pt")
        inputs.update(tokenized)
        if isinstance(inputs.get("mm_token_type_ids"), list):
            inputs["mm_token_type_ids"] = torch.tensor(inputs["mm_token_type_ids"], dtype=torch.long)
        if profile:
            timings["processor_ms"] = (time.perf_counter() - phase) * 1000
            token_count = int(inputs["input_ids"].numel())
            grid = inputs.get("image_grid_thw")
            grid_count = int(grid.prod(dim=-1).sum().item()) if isinstance(grid, torch.Tensor) else None
            phase = time.perf_counter()
        inputs = inputs.to(self.device)
        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["h2d_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
        hidden = self.model(**inputs).last_hidden_state[:, -1]
        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["forward_ms"] = (time.perf_counter() - phase) * 1000
            phase = time.perf_counter()
        if self.config.get("score_dtype") == "float32":
            score = float(torch.sigmoid(torch.nn.functional.linear(hidden.float(), self.score_head.weight.float())).item())
        else:
            score = float(torch.sigmoid(self.score_head(hidden)).item())
        if profile:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            timings["head_ms"] = (time.perf_counter() - phase) * 1000
            timings["total_ms"] = (time.perf_counter() - started) * 1000
            with Image.open(image_path) as source:
                source_size = list(source.size)
            print("qwen_profile " + json.dumps({
                **{name: round(value, 3) for name, value in timings.items()},
                "token_count": token_count,
                "grid_count": grid_count,
                "query_size": list(query.size),
                "source_size": source_size,
                "source_bytes": image_path.stat().st_size,
            }, separators=(",", ":")), flush=True)
        return score


def card_text(card: dict[str, Any], fields: list[str]) -> str:
    return "\n".join(f"{field}: {card[field]}" for field in fields if card.get(field))
