from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
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

    def query_view(self, image: Image.Image) -> Image.Image:
        name = self.config["reranker_query_views"][0]
        if name == "whole":
            return image
        if name == "label90":
            width, height = image.size
            return image.crop((int(width * 0.05), int(height * 0.18), int(width * 0.95), int(height * 0.94)))
        raise ValueError(f"unsupported reranker query view: {name}")

    def _messages(self, query: Image.Image, text: str, image_path: Path) -> list[dict[str, Any]]:
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
                    {"type": "image", "image": f"file://{image_path}", "min_pixels": self.config["min_pixels"], "max_pixels": self.config["max_pixels"]},
                    {"type": "text", "text": text},
                ],
            },
        ]

    @torch.inference_mode()
    def score(self, query: Image.Image, text: str, image_path: Path) -> float:
        messages = self._messages(query, text, image_path)
        prompt = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            [messages], image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )
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
        inputs = inputs.to(self.device)
        hidden = self.model(**inputs).last_hidden_state[:, -1]
        if self.config.get("score_dtype") == "float32":
            return float(torch.sigmoid(torch.nn.functional.linear(hidden.float(), self.score_head.weight.float())).item())
        return float(torch.sigmoid(self.score_head(hidden)).item())


def card_text(card: dict[str, Any], fields: list[str]) -> str:
    return "\n".join(f"{field}: {card[field]}" for field in fields if card.get(field))
