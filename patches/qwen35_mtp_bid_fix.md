# qwen35_mtp_bid_fix.patch

## What this patches

`conversion/qwen.py` in llama.cpp, class `_Qwen35MtpMixin.modify_tensors()`. Adds one assignment (`bid = bid + n_layer`) after the MTP-tensor rename, so the bid argument passed to downstream `Qwen2MoeModel.modify_tensors()` matches the renamed tensor's layer slot.

## Why it's required

Without this fix, converting a multimodal Qwen3.5-MoE NVFP4 source (architecture `Qwen3_5MoeForConditionalGeneration`, tensors arriving under `model.language_model.layers.{N}.*`) crashes with:

```
File "conversion/qwen.py", line 130, in modify_tensors
    datas.append(self._experts[bid][ename])
KeyError: 'model.layers.0.mlp.experts.0.down_proj.weight'
```

Root cause: the MTP mixin renames the tensor name from `mtp.layers.0.*` to `model.layers.48.*` but passes the original `bid=0` to the downstream MoE expert-stacker. The stacker uses `bid` as both a cache key (`self._experts[bid]`) and a name template (`f"model.layers.{bid}.mlp.experts.{xid}..."`), so the cache gets populated with layer-48 names under bid=0, then the stacker tries to look up layer-0 names in a cache that was drained during base-layer-0 processing earlier in the run. KeyError.

Full bug analysis with traceback walk: see [Technical Report §2.2](../paper/REPORT.md#22-bug-analysis).

## Why this doesn't manifest on text-only sources

For `Qwen3_5MoeForCausalLM` (text-only) Qwen3.5-MoE GGUF builds (e.g. Unsloth's MTP releases), tensors arrive under `model.layers.{N}.*` directly. The order in which the MoE expert-stacker fills and drains `self._experts[0]` happens to complete before the MTP path runs, so the same bug doesn't surface. The strip-prefix step in the multimodal path changes that ordering and the failure surfaces.

## Applies to

- **File:** `conversion/qwen.py`
- **Class:** `_Qwen35MtpMixin`
- **Method:** `modify_tensors()`
- **Anchor commit:** `0253fb21f` (llama.cpp build `b9187`, post-PR-22673 merge)
- **Affected sources:** Any `Qwen3_5MoeForConditionalGeneration`-architecture model converted via `convert_hf_to_gguf.py`. Includes the source weights used for this methodology's first release: [txn545/Qwen3.5-122B-A10B-NVFP4](https://huggingface.co/txn545/Qwen3.5-122B-A10B-NVFP4).

## How to apply

```bash
cd llama.cpp
git checkout 0253fb21f   # or any post-PR-22673 master commit
git apply /path/to/qwen35_mtp_bid_fix.patch
```

If `git apply` rejects due to upstream drift in surrounding code, the patch is small enough to apply manually: open `conversion/qwen.py`, find `_Qwen35MtpMixin.modify_tensors()`, and add the `bid = bid + n_layer` assignment immediately after the `name = name.replace(...)` line in the `if name.find("layers.") != -1:` branch.

## Upstream status

**Filed as [ggml-org/llama.cpp#23237](https://github.com/ggml-org/llama.cpp/pull/23237)** (2026-05-17, awaiting review). Branch `fix/qwen35-mtp-multimodal-bid-coherence` on `bit-incarnas/llama.cpp`. Drop this patch when the PR merges + the next methodology release builds off a post-fix mainline commit (see "Drop trigger" below).

## Drop trigger

Drop this patch when:

1. An equivalent fix lands in llama.cpp upstream master, AND
2. The next release built off a post-fix commit no longer needs the patch applied.

When the upstream PR merges, update this `.md` with the merge commit + the llama.cpp build that first includes the fix, and mark the patch as `superseded` in the methodology repo's README releases table.

## Cross-reference

This patch is also documented inline in:

- The model card README at [Incarnas/Qwen3.5-122B-A10B-NVFP4-MTP-GGUF](https://huggingface.co/Incarnas/Qwen3.5-122B-A10B-NVFP4-MTP-GGUF) (§"Source and conversion")
- The technical report at [`paper/REPORT.md` §2.2](../paper/REPORT.md#22-bug-analysis) and §7.2 (Reproduction recipe)

This file is the canonical machine-applicable diff source. The inline docs are summary forms.
