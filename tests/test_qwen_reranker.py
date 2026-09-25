import json
from types import SimpleNamespace

from PIL import Image

from wine_api import qwen_reranker


def test_disables_cudnn_sdp_before_model_load_and_defaults_preload_off(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "model.safetensors").touch()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "model_id": "org/model",
        "model_safetensors_sha256": "expected",
        "device": "cpu",
        "dtype": "float32",
    }))
    calls = []
    monkeypatch.setenv("WINE_DISABLE_CUDNN_SDP", "1")
    monkeypatch.delenv("WINE_RERANKER_PRELOAD", raising=False)
    monkeypatch.setattr(qwen_reranker, "sha256", lambda path: "expected")
    monkeypatch.setattr(qwen_reranker.torch.backends.cuda, "enable_cudnn_sdp", lambda enabled: calls.append(("sdp", enabled)))

    class Model:
        def eval(self):
            return self

    class LM:
        model = Model()
        lm_head = SimpleNamespace(weight=qwen_reranker.torch.tensor([[1.0], [0.0]]))

        def to(self, _):
            return self

    def load(*args, **kwargs):
        calls.append(("load", None))
        return LM()

    monkeypatch.setattr(qwen_reranker.Qwen3VLForConditionalGeneration, "from_pretrained", load)
    tokenizer = SimpleNamespace(get_vocab=lambda: {"yes": 0, "no": 1})
    monkeypatch.setattr(qwen_reranker.AutoProcessor, "from_pretrained", lambda *_args, **_kwargs: SimpleNamespace(tokenizer=tokenizer))

    reranker = qwen_reranker.QwenReranker(model_path, config_path)

    assert calls == [("sdp", False), ("load", None)]
    assert not reranker.preload_enabled


def test_reranker_preloads_references_and_resizes_query_once(tmp_path, monkeypatch):
    calls = []

    def resize(element, image_patch_size):
        calls.append((element["image"], image_patch_size))
        if isinstance(element["image"], Image.Image):
            return element["image"].resize((32, 32))
        with Image.open(element["image"][7:]) as image:
            return image.convert("RGB").resize((32, 32))

    monkeypatch.setattr(qwen_reranker, "fetch_image", resize)
    reranker = qwen_reranker.QwenReranker.__new__(qwen_reranker.QwenReranker)
    reranker.config = {"min_pixels": 16, "max_pixels": 1024, "reranker_query_views": ["whole"], "instruction": "same"}
    reranker.preload_enabled = True
    reranker.reference_images = {}
    path = tmp_path / "reference.png"
    Image.new("RGB", (64, 48), "red").save(path)

    reranker.preload_references([path])
    query = Image.new("RGB", (48, 64), "blue")
    resized_query = reranker.query_view(query)
    messages = reranker._messages(resized_query, "wine", path)

    assert calls == [(f"file://{path.resolve()}", 16), (query, 16)]
    assert messages[1]["content"][4]["image"] is reranker.reference_images[path.resolve()]
    reranker.close()

    reranker.preload_enabled = False
    reranker.preload_references([path])
    assert not reranker.reference_images
    assert reranker.query_view(query) is query
    assert reranker._messages(query, "wine", path)[1]["content"][4]["image"] == f"file://{path}"
    assert calls == [(f"file://{path.resolve()}", 16), (query, 16)]
    resized_query.close()
    query.close()
