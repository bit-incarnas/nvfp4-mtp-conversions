# NVFP4 + MTP Conversions

Methodology repo for Incarnas's NVFP4 GGUF conversions with MTP (Multi-Token Prediction) head retention for self-speculative decoding via llama.cpp on Blackwell-class hardware.

This repo carries the methodology toolchain: the technical report, the local converter patches required on top of mainline llama.cpp, the bench scripts + config + raw bench output cited from the report, and the plots derived from that data. Per-release GGUF model artifacts live on HuggingFace under [Incarnas](https://huggingface.co/Incarnas) and link back here for methodology + bench provenance.

## Scope

The work covered by this methodology:

- NVFP4-quantized GGUF conversion from NVIDIA ModelOpt source weights (the `modelopt` quant-method path; not the `compressed-tensors` flavor)
- Retention of the MTP head for self-speculative decoding via llama.cpp's `--spec-type draft-mtp` path
- Blackwell-native NVFP4 runtime via `BLACKWELL_NATIVE_FP4=1`
- Hybrid Mamba-MoE architecture handling (Qwen3.5-MoE family today; broader as the family grows)
- Bench validation under chat-completions discipline per the [chat-vs-raw methodology](https://github.com/bit-incarnas/chat-vs-raw-methodology) project

## What's here

```
.
├── paper/                  -- the technical report (markdown source + pandoc-built PDF)
├── patches/                -- local converter patches required on top of mainline llama.cpp
├── scripts/                -- conversion + bench wrappers referenced from the report
├── configs/                -- llama-server / lm-eval / converter configs referenced from the report
├── benchmark_results/      -- raw bench output the report cites (JSON / CSV)
└── plots/                  -- source for figures referenced in the report
```

The technical report at [`paper/REPORT.md`](paper/REPORT.md) (PDF: [`paper/REPORT.pdf`](paper/REPORT.pdf)) is the canonical methodology document. Every released GGUF model card links back to it for methodology + bench reproducibility.

## Releases under this methodology

| Release | HuggingFace | Status | Notes |
| :--- | :--- | :--- | :--- |
| `Qwen3.5-122B-A10B-NVFP4-MTP-GGUF` | [Incarnas/Qwen3.5-122B-A10B-NVFP4-MTP-GGUF](https://huggingface.co/Incarnas/Qwen3.5-122B-A10B-NVFP4-MTP-GGUF) | v1.0 (2026-05-16) | First release. AR + MTP (`--spec-type draft-mtp --spec-draft-n-max 2`). +46% long-decode throughput vs AR on Blackwell Pro 96 GiB. Requires `patches/qwen35_mtp_bid_fix.patch` |

## Patches

Local patches required on top of mainline llama.cpp for the current methodology release set. Each patch ships with a sibling `.md` carrying the rationale, upstream PR status, and the drop trigger that retires the patch.

| Patch | Applies to | Upstream status |
| :--- | :--- | :--- |
| [`patches/qwen35_mtp_bid_fix.patch`](patches/qwen35_mtp_bid_fix.patch) | llama.cpp `0253fb21f` (build `b9187`) -- pre-merge of fix | **Merged** as [ggml-org/llama.cpp#23237](https://github.com/ggml-org/llama.cpp/pull/23237) in master commit [`1867a0c69`](https://github.com/ggml-org/llama.cpp/commit/1867a0c6923eaebb7a53965f6cdbc0ace55142a3) on 2026-05-18; patch retires when next methodology release builds off a post-fix mainline commit -- see [`qwen35_mtp_bid_fix.md`](patches/qwen35_mtp_bid_fix.md) |

## Bench environment

Reference rig for all numbers in the report:

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GiB GDDR7, 300 W TDP)
- **CPU:** AMD Ryzen Threadripper 7960X (24 cores / 48 threads)
- **RAM:** 128 GiB DDR5 ECC RDIMM @ 6400 MHz
- **OS:** Linux 7.0.0-15-generic with NVIDIA driver supporting `BLACKWELL_NATIVE_FP4`
- **llama.cpp:** locally-built from documented commits per release

Bench scripts are simple bash + python over `urllib.request` against the llama-server HTTP endpoint; no vendor harness dependencies.

## License

MIT. See [`LICENSE`](LICENSE).

## Citation

```bibtex
@techreport{incarnas_2026_nvfp4_mtp_conversions_qwen35_122b,
  author       = {Incarnas},
  title        = {{Qwen3.5-122B-A10B-NVFP4-MTP-GGUF Technical Report}},
  institution  = {Independent},
  year         = {2026},
  month        = {May},
  note         = {v1.0; companion to https://huggingface.co/Incarnas/Qwen3.5-122B-A10B-NVFP4-MTP-GGUF},
  url          = {https://github.com/bit-incarnas/nvfp4-mtp-conversions/blob/main/paper/REPORT.pdf}
}
```

## Attribution

- llama.cpp by Georgi Gerganov and contributors: <https://github.com/ggml-org/llama.cpp>
- NVIDIA ModelOpt: NVFP4 source-weight toolchain
- Base models: per-release model card; first release derives from [Qwen/Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) via [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4)
- Sibling methodology: [bit-incarnas/chat-vs-raw-methodology](https://github.com/bit-incarnas/chat-vs-raw-methodology) -- bench-mode discipline applied to every release here
