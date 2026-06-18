# Step 4 handoff — paste into next Claude prompt, then delete

**Branch:** `uProfAnalysis` | **Scope:** operator-level only, avg corner, no model claims
**Status:** Step 4 mapping **VALIDATED 2026-06-17** (primary-CSV regen + §3 diff + Chris sign-off). Next: paper scaffold.

> Regenerated from primary `results/` CSVs via `--source primary`; §3-vs-primary energy-layer diff 83/83 numeric PASS + 1 expected-N/A, 2026-06-17. Chris methodological sign-off 2026-06-17.

---

## Read first (30 s)

| What | Where |
|------|-------|
| Durable project truth | `PROJECT_CONTEXT.md` §6–§7 |
| Energy numbers (committed snapshot) | `ANALYSIS_REFERENCE.md` §3 — **21 ops**, real J/op + ratios (transcription source; validated vs primary) |
| Step 4 full table + prose | `STEP4_OPERATOR_ENGINE_MAPPING.md` |
| Step 4 machine output | `results/step4_operator_engine_mapping.csv` |
| Repro script | `step4_operator_engine_mapping.py` |

**PROVENANCE:** Step 4 energies regenerated from primary tower `results/` CSVs (`--source primary`); §3-vs-primary energy-layer diff 83/83 numeric PASS + 1 expected-N/A (2026-06-17). `ANALYSIS_REFERENCE.md` §3 unchanged — diff harness confirms transcription fidelity.

---

## Sessions (tower)

| Session | Engines | Idle |
|---------|---------|------|
| `20260612_143854` | cpu/igpu FP32, npu XINT8 | 216.22 J |
| `20260615_145614_cpu_int8` | cpu INT8 control | 326.49 J |

Decomposition identity `precision × arch = gap` → 1.000 across 20 triplet ops (consistency check only).

---

## Step 4 router (mechanism → metric)

| Bucket | Ops | Rank on | Candidates |
|--------|-----|---------|------------|
| **COMPUTE_BOUND_DENSE** | GEMMs, dense convs, `attn_block_fused`, `sra_conv2d` | matched INT8 | cpu_INT8 vs npu_INT8 (iGPU out) |
| **MEMORY_BOUND** | softmax, gelu, norms, pool, residual, **depthwise_conv2d** | FP32 | cpu_FP32 vs igpu_FP32 (NPU out) |

`PRECISION_GUARD=0.25` advisory only — never routes.
`TIE_THRESHOLD=0.10`, `TIE_POLICY=report` — Chris-approved 2026-06-17.
`SIGN_MARGIN=0.10` — raw headline≠best-INT8 flips within margin are guarded out — Chris-approved 2026-06-17.

Contested-middle ties (TIE_THRESHOLD=0.10) are sensitive to tensor dimensions; the mapping is single-corner (avg, N=197, D=768). Parameter sweeps over batch size and sequence length could change which engine wins a tied cell — flagged as future work, per supervisor sign-off.

---

## Winners at a glance (avg corner)

**NPU (matched INT8):** all trusted GEMMs + `downsample_conv2d`, `patch_embed_conv2d` (arch 3–14×, precision < 1 on dense).

**CPU (matched INT8, contested):** `attn_score_matmul`, `attn_value_matmul`, `xcit_cov_matmul`.

**CPU (FP32):** `softmax`, `gelu`, `layer_norm`, `batch_norm`, `residual_add`.

**iGPU (FP32):** `avg_pool_token_mixer`.

**Ties (FP32, ≤10%):** `depthwise_conv2d` tie(cpu/igpu), `group_norm` tie(cpu/igpu).

**Excluded / N/A:** `sra_conv2d×npu` UNTRUSTED; `attn_block_fused×npu` N/A.

---

## Two divergence columns — do not conflate

| Column | Meaning |
|--------|---------|
| **`sign_divergence`** | Step-3 trap (SIGN_MARGIN guarded): headline winner ≠ best-INT8 winner. **True:** `softmax[s1]`, `depthwise_conv2d` **ONLY**. N/A: sra, attn_block_fused, avg_pool. |
| **`diverges_deployment`** | Step-4 trap: deploy winner ≠ naive production-grid argmin. **True:** `batch_norm`, `group_norm`. **False:** softmax, depthwise (tie or match). N/A: EXCLUDED / no matched rec. |

**Marginal flip (guarded out):** `batch_norm` — raw headline≠best-INT8 but gap=1.03 within SIGN_MARGIN; robust best-INT8=CPU (arch=0.81). Not in divergence pair.

**Paper headline pair:** softmax + depthwise — INT8-vs-FP32 attribution trap, separate from FP32 deployment pick.

---

## Trust carry-forwards (always apply)

- `sra_conv2d × npu` → UNTRUSTED
- `attn_block_fused × npu` → N/A
- `avg_pool_token_mixer × cpu_int8` → LOW-TRUST (irrelevant to FP32 routing)

---

## Deferred / not done

- [x] Step 4 regen from **tower** `results/` CSVs (paper-grade) — **done 2026-06-17**
- [x] Chris sign-off on methodology + tie/sign-margin policy — **done 2026-06-17**
- [ ] Paper scaffold (intro / methods / results)
- [ ] small/large SDPA corners; Track-2 shape fixes; model-level work
- [ ] `uProfBenchmarking`: idle re-run + sra remediation

---

## Do not touch unless asked

`PROJECT_CONTEXT.md`, `ANALYSIS_REFERENCE.md`, harness/pipeline/session files, tower `results/` artifacts.
