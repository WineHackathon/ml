# Optimization summary

## Promoted runtime

The promoted local runtime is the frozen v2 SigLIP2 gallery, label-blind OCR entity fusion into a six-item pool, and sequential `Qwen/Qwen3-VL-Reranker-2B` scoring. It was tested at `http://127.0.0.1:8080`; `WINE_RERANKER=off` is the same-gallery SigLIP-only rollback.

| Evidence set | Scope | Top-1 | Top-5 | Errors | Full HTTP latency |
| --- | --- | ---: | ---: | ---: | --- |
| OLD DEV | tuned silver, 12 exact | 12/12 | 12/12 | 0 | median 8.092 s, p95 8.921 s, max 9.321 s |
| holdout v1 | one-shot silver, now historical | 8/11 | 11/11 | 0 | median 7.970 s, max 9.193 s |
| holdout v2 | sealed selection validation | **5/5** | **5/5** | 0 | median 7.031 s, max 8.721 s |
| organizer client | unlabeled protocol/latency only | 3/3 valid | n/a | 0 | 7.861–8.171 s |

These tiny agent-reviewed sets are evidence, not a population accuracy estimate. Holdout v2 contains only five exact scenes. No claim of local or global optimality is warranted.

## Frozen identity

- portable v2 index: `artifacts/catalog_v2/catalog_index.npz`, SHA-256 `8a884e6eedb8212d22c5647f23bb63bb6c89035eb9e229f16ef7278f89e7a3b8`
- sanitized v2 manifest: `data/catalog_manifest.jsonl`, SHA-256 `0cbf9ac25a4d4f6cfd0911e70c693f24458a12785e104f09dae8910e88211bff`
- source index before path-only metadata rebase: SHA-256 `52f3c83a8476c5a7ee27867f6aefa7f402477f970c3cac410edc61a65d567e70`; embeddings SHA-256 remains `56f357d25f1445965d1f88cc318612a880f720d58b832327f7d50f01f8d27598`
- selected runtime config: `artifacts/qwen_reranker/config.json`, SHA-256 `b2ce7c445029234628261f13d8aad2e57ca30d65bb7bfb0efe54bf9b43ae9ea4`
- Qwen weights: revision `4bd860ac4f15ad1897a214615cccc700f8f71818`, SHA-256 `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`

## Launch and rollback

```bash
./scripts/run.sh
```

The launcher resolves every path from the checkout and selects the promoted SigLIP2 + OCR fusion + Qwen 2B pipeline. For immediate visual-only rollback, run `WINE_RERANKER=off ./scripts/run.sh`; OCR and Qwen are then not loaded.

## Exhausted bounded branches

- Five independently sourced catalog repairs and the broader v2 source recovery increased exact gallery coverage to 12/12 DEV targets; v2 contains 2,088/2,103 slugs. Fifteen catalog rows remain unresolved.
- Fixed OCR reranking did not improve the original proxy; the later OCR stage is used only for general entity-based candidate admission/tie evidence before Qwen.
- The frozen diagonal retrieval head reduced top-5 and was rejected.
- DINOv2 whole/center and label-ROI retrieval produced 0/12 top-5 and were rejected after implementation checks passed.
- Qwen 2B batch-5 changed BF16 scores and regressed one exact case without reducing latency; sequential scoring is frozen.
- Qwen 4B listwise reached 10/12 OLD DEV. Qwen 8B Torch reached 11/12 but failed the live 10-second limit.
- MLX 8B 4-bit/full-pixel reached 12/12 OLD DEV and 9/11 reused v1, but the official client had a 10.011-second timeout. The only authorized lower-reference-pixel variant reduced latency but regressed OLD DEV to 10/12; it was rejected.
- Float32 rescoring of the 2B head removed BF16 quantized ties but reduced OLD DEV to 11/12; the promoted behavior keeps the independently frozen OCR evidence tie-break.

## Remaining ceilings

- The 3-second target is not met; accuracy took priority under the hard 10-second contract.
- Most catalog joins remain provisional. The official-source audit stopped after 118/120 checked routes showed the same assets and two returned 404; 1,771 routes remain unaudited because the bounded sample found no changed asset.
- Some residual v1 disagreements are likely aliases, duplicate catalog cards, redesigned labels, or doubtful silver truth. Resolving them requires authoritative SKU identity and new product views, not more tuning on reused labels.
- Future accuracy work needs more genuinely independent real scenes, especially redesigned/vintage sibling labels and multi-bottle frames. Existing DEV and both holdouts must not be presented as fresh evidence again.
