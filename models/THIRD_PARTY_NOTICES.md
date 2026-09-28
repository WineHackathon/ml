# Bundled model weights

Pinned upstream snapshots and their model-card license declarations:

- [`google/siglip2-base-patch16-384`](https://huggingface.co/google/siglip2-base-patch16-384), revision `f775b65a79762255128c981547af89addcfe0f88` — Apache-2.0.
- [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B), revision `4bd860ac4f15ad1897a214615cccc700f8f71818` — Apache-2.0.
- [`PaddlePaddle/PP-OCRv5_mobile_det`](https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det), revision `0d63e78e2b680928f6b1747d76a08db6e645efb7` — Apache-2.0.
- [`PaddlePaddle/eslav_PP-OCRv5_mobile_rec`](https://huggingface.co/PaddlePaddle/eslav_PP-OCRv5_mobile_rec), revision `7553801264d3379d8d2e854971989e5e26c22e03` — Apache-2.0.

Model files are distributed unchanged. Qwen's single 4.26 GB SafeTensors file is split into Git LFS objects for GitHub's per-file limit, then reassembled byte-for-byte by `scripts/download_models.py` using the hashes in `models.lock.json`.
