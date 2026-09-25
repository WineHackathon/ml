# Optimization summary

## Runtime profiles

Both tested profiles use the frozen v2 SigLIP2 gallery, label-blind OCR entity fusion into a six-item pool, and sequential `Qwen/Qwen3-VL-Reranker-2B` scoring. `scripts/run.sh` is the Apple M4 Pro MPS + CPU-OCR profile. `scripts/run-h100.sh` is the one-H100 profile with full-FP32 Qwen, cached references, cuDNN SDPA disabled, and a persistent GPU OCR worker. Both expose the same HTTP API; `WINE_RERANKER=off` is the same-gallery SigLIP-only rollback.

| Evidence set | Scope | Top-1 | Top-5 | Errors | Full HTTP latency |
| --- | --- | ---: | ---: | ---: | --- |
| OLD DEV | tuned silver, 12 exact | 12/12 | 12/12 | 0 | median 8.092 s, p95 8.921 s, max 9.321 s |
| holdout v1 | one-shot silver, now historical | 8/11 | 11/11 | 0 | median 7.970 s, max 9.193 s |
| holdout v2 | sealed selection validation | **5/5** | **5/5** | 0 | median 7.031 s, max 8.721 s |
| organizer client | unlabeled protocol/latency only | 3/3 valid | n/a | 0 | 7.861–8.171 s |
| H100 full-28 silver | consumed labels | 12/12 DEV; 8/11 v1; 5/5 v2 | 12/12 DEV; 11/11 v1; 5/5 v2 | 0 | median 2.157 s, p95 3.016 s, max 4.251 s |
| H100 field20 | unlabeled latency sample | n/a | n/a | 0 | median 2.113 s, p95 2.677 s, max 3.237 s |

The silver labels are tiny, agent-reviewed, and already consumed during development; they are not organizer ground truth or a population accuracy estimate. Holdout v2 contains only five exact scenes. The separate frozen three-request protocol check reproduced 3/3 expected slugs, each under 10 seconds, but is parity evidence rather than an accuracy set. No claim of local or global optimality is warranted.

The H100 field sample establishes p95 below three seconds for that run, not that every request finishes below three seconds: its maximum was 3.237 seconds. The label-backed run also reached 4.251 seconds. Treat three seconds as a measured sample target, not a demonstrated production SLO. Cold readiness was 159–164 seconds, and the API plus OCR worker used about 7.8–8.0 GiB RSS.

## Frozen identity

- portable v2 index: `artifacts/catalog_v2/catalog_index.npz`, SHA-256 `bcf7caccccf1629863794313c5c90efbf2de5be890917c7f69d77bb761390fce`
- sanitized v2 manifest: `data/catalog_manifest.jsonl`, SHA-256 `be1865f14c13b25a2a2b2223402650dd4164163e43ed8561b7a2fbfe5955fd8c`
- pre-quarantine source index: SHA-256 `8a884e6eedb8212d22c5647f23bb63bb6c89035eb9e229f16ef7278f89e7a3b8`; quarantine removed two gallery rows for one slug, while every unaffected entry and embedding row remained unchanged
- MPS config: `artifacts/qwen_reranker/config.json`, SHA-256 `217e76a35bcac995efc53fbee894a02dbaea7460a985d1c5c7df78a524edc009`
- H100 FP32 config: `artifacts/qwen_reranker/config-h100.json`, SHA-256 `5980e6f24598e425e754e886bda247859857cddf6d819f16c511d36bf5d09abe`
- Qwen weights: revision `4bd860ac4f15ad1897a214615cccc700f8f71818`, SHA-256 `466ec01961061e9d7f804b4fb1625fb6f406106cd1567e026096d4736fa9d5b9`

## Launch and rollback

```bash
./scripts/run.sh
```

The local launcher resolves every path from the checkout. The H100 launcher requires an external model root:

```bash
export WINE_MODELS_ROOT=/path/to/persistent/models
./scripts/setup-h100.sh
./scripts/run-h100.sh
```

For immediate visual-only rollback, run `WINE_RERANKER=off ./scripts/run.sh`; OCR and Qwen are then not loaded. The H100 path is native Linux only; CUDA Docker was not verified.

## Exhausted bounded branches

- Five independently sourced catalog repairs and the broader v2 source recovery increased exact gallery coverage. A later audit proved that the official dry Citron/Chardonnay route served an image for a different wine. That image was quarantined, reducing the index to 2,087/2,103 slugs while preserving the product metadata; no verified replacement was found.
- Fixed OCR reranking did not improve the original proxy; the later OCR stage is used only for general entity-based candidate admission/tie evidence before Qwen.
- The frozen diagonal retrieval head reduced top-5 and was rejected.
- DINOv2 whole/center and label-ROI retrieval produced 0/12 top-5 and were rejected after implementation checks passed.
- Qwen 2B batch-5 changed BF16 scores and regressed one exact case without reducing latency; sequential scoring is frozen.
- Qwen 4B listwise reached 10/12 OLD DEV. Qwen 8B Torch reached 11/12 but failed the live 10-second limit.
- MLX 8B 4-bit/full-pixel reached 12/12 OLD DEV and 9/11 reused v1, but the official client had a 10.011-second timeout. The only authorized lower-reference-pixel variant reduced latency but regressed OLD DEV to 10/12; it was rejected.
- Float32 rescoring of the 2B head removed BF16 quantized ties but reduced OLD DEV to 11/12; the promoted behavior keeps the independently frozen OCR evidence tie-break.

## Remaining ceilings

- The Mac profile does not meet the 3-second target. The H100 field20 run met p95 below three seconds, but its maximum exceeded three seconds; production SLO compliance is not established.
- Most catalog joins remain provisional. An earlier bounded official-source audit found 118/120 checked routes served the same assets and two returned 404. The later Citron/Chardonnay finding proved that route agreement alone is not identity proof; the remaining catalog is not fully identity-audited.
- Some residual v1 disagreements are likely aliases, duplicate catalog cards, redesigned labels, or doubtful silver truth. Resolving them requires authoritative SKU identity and new product views, not more tuning on reused labels.
- Future accuracy work needs more genuinely independent real scenes, especially redesigned/vintage sibling labels and multi-bottle frames. Existing DEV and both holdouts must not be presented as fresh evidence again.
