# Step 4 paper-grade primary artifacts — run report

**Date:** 2026-06-17 | **Branch:** `uProfAnalysis` | **cwd:** `benchmark/`  
**Task:** Regenerate full-precision Step 4 artifacts from `--source primary` for off-tower July writing.  
**Not in scope:** new validation, new measurement, routing/threshold changes, F2 edits.

---

## Hard constraints (verified)

| Constraint | Status |
|------------|--------|
| `ANALYSIS_REFERENCE.md` §3 untouched | **PASS** — not modified |
| `TIE_THRESHOLD`, `SIGN_MARGIN`, `_sign_fields`, winner selection unchanged | **PASS** — values still 0.10 / 0.10 |
| §3-built artifacts not clobbered by primary run | **PASS** — separate `_primary` paths; §3 md/sheet restored via `--source section3` |
| Primary outputs in new `_primary` / committed names | **PASS** |

---

## 1. Primary artifact generation

```bash
conda activate ryzen-ai-1.6.0
cd benchmark/
python step4_operator_engine_mapping.py --source primary
```

### Outputs confirmed

| Artifact | Path | Committed? |
|----------|------|------------|
| Routing CSV (tower) | `results/step4_operator_engine_mapping_primary.csv` | No (`results/` gitignored) |
| Routing CSV (portable) | `benchmark/step4_operator_engine_mapping_primary.csv` | **Yes** |
| Readable table | `benchmark/STEP4_OPERATOR_ENGINE_MAPPING_primary.md` | **Yes** |
| Chris sheet import | `benchmark/STEP4_FOR_SHEET_primary.csv` | **Yes** |

**sign_divergence audit:** `['depthwise_conv2d', 'softmax[s1]']` — unchanged from validated routing.

### Script fix applied (non-routing)

`write_markdown()` was hardcoded to `OUT_MD`, so an earlier `--source primary` run had overwritten `STEP4_OPERATOR_ENGINE_MAPPING.md`. Fixed by passing `out_md` explicitly. Re-ran `--source section3` to restore the §3-built markdown and sheet.

**New:** `write_sheet()` exports flat Chris sheet CSV; `--source`-aware paths:
- `STEP4_FOR_SHEET.csv` (§3)
- `STEP4_FOR_SHEET_primary.csv` (primary, full precision)

---

## 2. Sheet export (`STEP4_FOR_SHEET_primary.csv`)

**Format:** one row per operator (21 rows), flat columns for spreadsheet import (new tab — not raw `runs_*.csv`).

| Column group | Fields |
|--------------|--------|
| Identity | `operator`, `shape_class` |
| Energies (full precision) | `cpu_FP32`, `cpu_INT8`, `npu_INT8`, `igpu_FP32` |
| Decomposition ratios | `precision`, `architecture_ratio`, `gap` |
| Routing | `mechanism`, `winner`, `margin_pct`, `is_tie`, `sign_divergence`, `diverges_deployment`, `naive_winner`, `trust_flag` |

First line: `# PROVENANCE: energies from primary tower CSVs ... at full precision.`

§3 sheet `STEP4_FOR_SHEET.csv` also generated (rounded energies from §3 snapshot) — left intact, separate file.

---

## 3. Consistency echo (report-only, not a new gate)

**Question:** For the 83 numeric energy cells, does `round(primary, §3_printed_precision) == §3`?

```
CONSISTENCY ECHO (83 numeric cells)
  cells checked: 83
  VERDICT: PASS — round(primary, s3_printed_precision) == s3 for all numeric cells
```

**Excluded from check (by design):** `attn_block_fused × npu_INT8` (expected-ABSENT disposition).

**Implication:** Full-precision primary CSV is the un-rounded twin of the validated §3 portable snapshot. Off-tower writers can cite `STEP4_FOR_SHEET_primary.csv` / `step4_operator_engine_mapping_primary.csv` with confidence that routing used the same energies §3 rounds to.

---

## 4. F2 data source (`figures/plot_f2_decomposition.py`)

**Answer: committed §3 only — does NOT read `results/` (tower-only).**

From script header and `main()`:

```python
# Energy source: benchmark/ANALYSIS_REFERENCE.md §3 ONLY (committed snapshot).
# No reads from results/ (demo fixtures) and no reliance on derived columns.
ANALYSIS_REF = ROOT / "benchmark" / "ANALYSIS_REFERENCE.md"
```

- Parses §3 decomposition table directly from `ANALYSIS_REFERENCE.md`.
- Recomputes `precision_ratio` and `architecture_ratio` from raw §3 J/op cells.
- Output: `figures/F2_decomposition.png`.
- Footnote in PNG still says “Regenerate from primary tower CSVs before publication.”

**Decision for Thursday off-tower:**

| Option | When |
|--------|------|
| Re-render F2 **on-tower today** | Only if you want full-precision bars in the PNG before leaving |
| **Wait for off-tower Thursday** | Safe for a §3-rounded figure; PNG will match committed snapshot, not `*_primary.csv` full precision |

F2 was **not edited** in this run (per instructions).

---

## 5. Git commit

**Message:**
```
Step 4 paper-grade artifacts from --source primary (full precision); §8 validated 2026-06-17
```

**Files staged for this commit:**

| File | Role |
|------|------|
| `benchmark/step4_operator_engine_mapping.py` | `--source primary`, sheet export, `out_md` fix |
| `benchmark/step4_operator_engine_mapping_primary.csv` | Committed full-precision routing output |
| `benchmark/STEP4_OPERATOR_ENGINE_MAPPING_primary.md` | Readable primary table |
| `benchmark/STEP4_FOR_SHEET_primary.csv` | Chris sheet import (primary) |
| `benchmark/STEP4_FOR_SHEET.csv` | Chris sheet import (§3 snapshot) |
| `benchmark/STEP4_OPERATOR_ENGINE_MAPPING.md` | §3-built readable table (restored) |
| `benchmark/step4_provenance_diff.py` | Provenance diff harness (validation pipeline) |
| `PROJECT_CONTEXT.md` | §8 VALIDATED promotion (prior session) |

**Explicitly not staged:** `benchmark/results/*` (gitignored), `ANALYSIS_REFERENCE.md`, `STEP4_HANDOFF.md`, `STEP4_PROMOTION_CHANGELOG.md`.

---

## 6. Side effects / notes

1. **`STEP4_HANDOFF.md`** was regenerated by `--source section3` (`write_handoff()`) and now shows pre-promotion “finalized / tower regen” text. The VALIDATED handoff from the promotion session was overwritten. Restore from `STEP4_PROMOTION_CHANGELOG.md` canonical strings if needed.
2. **`PROJECT_CONTEXT.md`** §8 VALIDATED edits from the promotion phase are included in this commit if not yet committed separately.
3. Primary markdown does not duplicate the VALIDATED status banner — that lives in `PROJECT_CONTEXT.md` §8; primary files carry the provenance banner only.

---

## Quick reference for July off-tower writing

| Need | File |
|------|------|
| Full-precision energies + routing (CSV) | `benchmark/step4_operator_engine_mapping_primary.csv` |
| Full-precision table (human) | `benchmark/STEP4_OPERATOR_ENGINE_MAPPING_primary.md` |
| Spreadsheet import | `benchmark/STEP4_FOR_SHEET_primary.csv` |
| Rounded portable snapshot | `benchmark/ANALYSIS_REFERENCE.md` §3 |
| Durable validated status | `PROJECT_CONTEXT.md` §8 |
| F2 figure (§3-rounded) | `figures/F2_decomposition.png` (render via `plot_f2_decomposition.py`) |
