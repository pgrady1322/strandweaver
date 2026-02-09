# StrandWeaver — Comprehensive Code Review & Ongoing Repairs

**Date:** 2026-02-07  
**Scope:** 42 Python files (8 packages), 15 Nextflow files, `user_training/` subsystem  
**Test baseline:** 44/44 tests passing before repairs began

---

## Table of Contents

1. [Missing Imports & Phantom Exports](#1-missing-imports--phantom-exports)
2. [Problematic Code Areas](#2-problematic-code-areas)
3. [Data Passing Between Phases](#3-data-passing-between-phases)
4. [Model Training Capability](#4-can-users-train-models)
5. [Priority Summary](#5-priority-summary)
6. [Repair Log](#6-repair-log)

---

## 1. Missing Imports & Phantom Exports

### 🔴 CRITICAL — `user_training/__init__.py` fatal import chain

`user_training/__init__.py` (lines 28–31) imports `TrainingDataGenerator` from
`config_based_workflow.py`, which at line 34 does:

```python
from training.synthetic_data_generator import (...)
```

The `training/` package **does not exist** in this repo — it only exists in
`strandweaver-dev`. Any code that does
`from strandweaver.user_training import TrainingDataGenerator` crashes with
`ModuleNotFoundError`.

**Impact:** The entire `user_training` package is un-importable.

### 🟡 HIGH — Phantom `ULReadMapper` in `assembly_core/__init__.py`

`assembly_core/__init__.py` line 80 lists `"ULReadMapper"` in `__all__` but
**never imports it**. It was renamed to `LongReadOverlay`; the `__all__` entry
is stale.

### 🟡 HIGH — `KmerGraph` inconsistent import path

`KmerGraph` is imported from `dbg_engine_module` at line 23 and exported in
`__all__`. It is also defined in other files. The canonical definition lives in
`dbg_engine_module`, but the duplication can confuse downstream imports.

### 🟢 LOW — `"assemble"` in `__all__` but raises `NotImplementedError`

`assembly_core/__init__.py` line 88 — The `assemble()` function is exported but
is a placeholder that raises `NotImplementedError`.

---

## 2. Problematic Code Areas

### 🔴 CRITICAL — `_run_ancient_pipeline` argument mismatch (TypeError on every call)

**Definition** (`pipeline.py` line 1389):

```python
def _run_ancient_pipeline(self, corrected_reads, ul_reads, ul_anchors, hic_data, ml_k_model)
```

Expects **5 positional args** (including `ul_anchors`).

**Call site** (`pipeline.py` line 658):

```python
self._run_ancient_pipeline(long_read_files, ul_read_files, hic_data, ml_k_model)
```

Passes **4 args**, omitting `ul_anchors` → `TypeError`.

### 🔴 CRITICAL — `_run_mixed_pipeline` same argument mismatch

**Definition** (`pipeline.py` line 1413):

```python
def _run_mixed_pipeline(self, corrected_reads, long_reads, ul_reads, ul_anchors, hic_data, ml_k_model)
```

Expects **6 positional args**.

**Call site** (`pipeline.py` line 663):

```python
self._run_mixed_pipeline(illumina_files, long_read_files, ul_read_files, hic_data, ml_k_model)
```

Passes **5 args**, omitting `ul_anchors` → `TypeError`.

### 🔴 CRITICAL — `--skip-profiling` + `--skip-correction` leaves assembly with zero reads

When both flags are set, `_step_correct` is skipped so
`self.state['corrected_files']` is never populated. `_step_assemble` (line 545)
builds file lists from `corrected_files` and gets empty lists, producing an
empty assembly.

**Fix needed:** Fall back to original (profiled or raw) input reads when
correction is skipped.

### 🟡 HIGH — `_calculate_hic_coverage` defined twice with incompatible return types

- First definition (line 1887): returns `float` (percentage).
- Second definition (line 2847): returns `Dict[int, float]` (per-node counts).

The second silently shadows the first. Any code expecting `float` gets `Dict`.

### 🟡 HIGH — `graph.edges` used as both list and dict

- Line 1903: `for edge in graph.edges:` — iterates keys (ints), not edge objects.
- Line 2862: `for edge_id, edge in graph.edges.items():` — correct dict iteration.

The first version's `hasattr(edge, 'edge_type')` on an int always returns
`False`, so Hi-C coverage is always 0.

### 🟡 HIGH — `--gpu-backend` CLI option silently ignored

CLI accepts `--gpu-backend` (line 213) and binds to `gpu_backend` parameter
(line 279), but the config builder only stores `use_gpu` and `gpu_device`.
`gpu_backend` is never written to config.

### 🟡 HIGH — `_save_graph` is a no-op

`pipeline.py` line 2940:

```python
def _save_graph(self, graph, output_path: Path):
    self.logger.debug(f"Graph export to {output_path} not yet implemented")
```

No GFA output is ever produced.

### 🟡 HIGH — Phantom `ULReadMapper` in `__all__`

`assembly_core/__init__.py` exports `"ULReadMapper"` but never imports it.
Explicit import → `ImportError`.

### 🟢 MEDIUM — `_read_file_streaming` doesn't handle `.gz`

Users who pass `.fq.gz` files (extremely common) get garbled binary data parsed
as FASTQ.

### 🟢 MEDIUM — All AI models load as `None` stubs ✅ FIXED (repair #11)

`_load_ai_models` (line 2957) sets every model to `None`. AI pipeline runs but
every model-dependent decision falls through to defaults.

### 🟢 MEDIUM — Checkpoint resume doesn't persist pipeline state ✅ FIXED (repair #8)

Checkpoint system saves/restores only the step name, not `self.state`. Resume
from `assemble` loses all prior outputs.

### 🟢 MEDIUM — Paired-end pairing lost during correction ✅ FIXED (repair #12)

Illumina R1/R2 files are merged into a flat list and lose their pairing
relationship through the correction step.

### 🟢 MEDIUM — `_step_classify_chromosomes` reads `hic_contact_map` never set ✅ FIXED (repair #13)

No pipeline step ever populates `self.state['hic_contact_map']`; it is always
`None`.

---

## 3. Data Passing Between Phases

### Direct Pipeline (Python `PipelineOrchestrator`)

| Step | Writes to state | Reads from state |
|------|----------------|-----------------|
| `_step_profile` | `error_profiles`, `technology_profiles` | — |
| `_step_correct` | `corrected_files` | `error_profiles` |
| `_step_assemble` | `graph`, `contigs` | `corrected_files` |
| `_step_finish` | `final_assembly` | `contigs`, `graph` |
| `_step_misassembly_report` | `misassembly_report` | `contigs`, `graph` |
| `_step_classify_chromosomes` | — | `graph`, `hic_contact_map` |

**Key issues:**

1. `hic_contact_map` — never populated by any step.
2. `corrected_files` — empty when `--skip-correction` is used (P0).
3. Checkpoint resume does not serialize `self.state` — all prior outputs lost.

### Nextflow Pipeline

**The Nextflow pipeline is non-functional end-to-end.**

1. **`extractEdges()` / `partitionGraph()` return empty lists**
   (`strandweaver.nf` line 232). Edge scoring and SV detection never receive
   input.

2. **9 missing CLI subcommands** invoked by Nextflow modules:
   `classify`, `kweaver`, `nf-build-graph`, `nf-edgewarden-filter`,
   `nf-pathweaver-iter-general`, `nf-threadcompass-aggregate`, `nf-strandtether-phase`,
   `nf-pathweaver-iter-strict`, `nf-export-assembly`.
   Only `pipeline`, `batch`, and `config` exist.

3. **Channel truthiness bug**: `if (ont_ul_reads)` / `if (hic_r1)` are always
   truthy in Nextflow DSL2 (channels are objects). UL and Hi-C branches always
   execute even when no data is provided.

4. **Only 5 of 17 processes have working CLI wiring**: `PROFILE_ERRORS`,
   `CORRECT_BATCH`, `MERGE_CORRECTED`, `MAP_UL_BATCH`, `ALIGN_HIC_BATCH`.

5. **Inconsistent invocation**: some modules use `strandweaver` entry point,
   others use `python3 -m strandweaver.cli`.

**Bottom line:** The Nextflow layer is scaffolding only. The direct Python
pipeline (`strandweaver pipeline`) is the only functional execution path.

---

## 4. Can Users Train Models?

### ❌ No — training is not functional in the release repo

| Component | Status | Details |
|-----------|--------|---------|
| `user_training/training_config.py` | ✅ Working | Dataclasses are complete and importable |
| `user_training/config_based_workflow.py` | 💀 Fatal | Imports `training.synthetic_data_generator` — doesn't exist |
| `user_training/generate_training_data.py` | 💀 Broken | Depends on broken workflow |
| `user_training/README.md` | ⚠️ Misleading | References classes that don't exist |
| `user_training/env_ml_training.yml` | ⚠️ Outdated | Python 3.8 (EOL), missing PyTorch/PyG |

**Per-model training status:**

| Model | Has Training Code? | Notes |
|-------|--------------------|-------|
| **KWeaver** | ❌ | Inference only (loads sklearn pickle) |
| **ErrorSmith** | ❌ | Classical only despite `BaseErrorClassifierAI` name |
| **EdgeWarden** | ⚠️ Partial | Has `train_all_technologies()` but no data pipeline |
| **PathWeaver GNN** | ❌ | Architecture defined (GATv2Conv) but no training loop |

**Missing from release repo** (exist in `strandweaver-dev` only):

- `training/synthetic_data_generator.py`
- `training/ml_training_system.py`
- `training/generate_ml_training_data.py`

---

## 5. Priority Summary

| Priority | Issue | Impact | Status |
|----------|-------|--------|--------|
| 🔴 P0 | `_run_ancient_pipeline` arg mismatch | TypeError on ancient runs | ✅ Fixed 2026-02-07 |
| 🔴 P0 | `_run_mixed_pipeline` arg mismatch | TypeError on mixed runs | ✅ Fixed 2026-02-07 |
| 🔴 P0 | `user_training` broken import chain | Package un-importable | ✅ Fixed 2026-02-07 |
| 🔴 P0 | Skip-correction = zero reads for assembly | Silent empty assembly | ✅ Fixed 2026-02-07 |
| 🟡 P1 | `_calculate_hic_coverage` shadowed | Dict vs float mismatch | ✅ Fixed 2026-02-08 |
| 🟡 P1 | `graph.edges` iterated as list | Hi-C coverage always 0 | ✅ Fixed 2026-02-08 |
| 🟡 P1 | `--gpu-backend` never stored | User option ignored | ✅ Fixed 2026-02-08 |
| 🟡 P1 | `_save_graph` is a no-op | No GFA output | ✅ Fixed 2026-02-08 |
| 🟡 P1 | Phantom `ULReadMapper` in `__all__` | ImportError | ✅ Fixed 2026-02-08 |
| 🟢 P2 | Nextflow pipeline non-functional | 9 missing subcommands | ✅ Fixed 2026-02-08 |
| 🟢 P2 | Checkpoint resume no state | Resume loses outputs | ✅ Fixed 2026-02-08 |
| 🟢 P2 | `.gz` not handled in streaming reader | Common format fails | ✅ Fixed 2026-02-08 |
| 🟢 P2 | All AI models are `None` stubs | AI features are no-ops | ✅ |
| 🟢 P2 | Paired-end pairing lost during correction | R1/R2 not recoverable after correction | ✅ Fixed 2026-02-09 |
| 🟢 P2 | `hic_contact_map` / `graph` never in state | Chromosome classification always gets None | ✅ Fixed 2026-02-09 |
| 🟢 P2 | `assemble()` stub in `assembly_core` | Public API raises NotImplementedError | ✅ Fixed 2026-02-09 |
| 🟢 P3 | No training code in release repo | Users cannot train | ⬜ |

---

## 6. Repair Log

*Repairs are logged here as they are applied.*

### 2026-02-07 — P0 Fixes

1. **`_run_ancient_pipeline` / `_run_mixed_pipeline` signature mismatch**
   (`pipeline.py` lines 1389–1435) — Removed the phantom `ul_anchors` parameter
   from both method signatures.  The call sites pass 4 args (ancient) / 5 args
   (mixed), and the delegation target `_run_hifi_pipeline` also has no
   `ul_anchors` parameter, so the definitions now match on both ends.

2. **`user_training` broken import chain** (3 files) —
   - `config_based_workflow.py`: Wrapped fatal
     `from training.synthetic_data_generator import ...` in `try/except`;
     sets `None` stubs when the `training` package is absent.
   - `config_based_workflow.py` `TrainingDataGenerator.__init__`: Now raises
     `RuntimeError` with clear install guidance when the backend is missing.
   - `user_training/__init__.py`: Guarded re-export with `try/except` so the
     package imports cleanly even without the backend.

3. **Skip-correction → zero reads** (`pipeline.py` `_step_assemble`) — Added
   fallback: when `corrected_files` is empty (e.g. `--skip-correction`) the
   pipeline now falls back to the raw input reads from
   `self.state['read_files']`, grouped by technology.  Also widened the tech
   matching to include `'hifi'` and `'ont_r10'` alongside `'ont'`/`'pacbio'`.

### 2026-02-08 — P1 Fixes

4. **`_calculate_hic_coverage` dual definition / `graph.edges` iteration bug**
   (`pipeline.py`) — The second definition (line ~2866) returned
   `Dict[int, float]` and silently shadowed the first (line ~1906) which
   returned `float`.  Additionally, both `_count_hic_edges` and the first
   `_calculate_hic_coverage` iterated `graph.edges` as a bare iterable (yielding
   dict keys / ints) instead of calling `.items()`.  Fixes:
   - Renamed second definition → `_calculate_hic_support_per_node`.
   - Fixed `.items()` iteration in `_count_hic_edges` and
     `_calculate_hic_coverage`.
   - Updated 2 BandageNG `export_for_bandageng` call sites (Illumina + HiFi
     pipelines) to call `_calculate_hic_support_per_node` (expects `Dict`).
   - 4 stat/logging call sites correctly keep `_calculate_hic_coverage` (expects
     `float`).

5. **`--gpu-backend` silently ignored** (`cli.py` line ~445) — `gpu_backend`
   was accepted by Click but never written to `pipeline_config`.  Now stored in
   `pipeline_config['hardware']['gpu_backend']`.  Also: passing an explicit
   backend (e.g. `--gpu-backend mps`) without `--use-gpu` now auto-enables GPU.

6. **Phantom `ULReadMapper` / `Anchor` in `assembly_core/__init__.py`** —
   - Replaced stale `"ULReadMapper"` in `__all__` with `"LongReadOverlay"`
     (the actual class name after rename).
   - Added `LongReadOverlay` import from `string_graph_engine_module`.
   - Added `Anchor` import from `dbg_engine_module` (was in `__all__` but
     never imported).
   - Removed duplicate `assemble()` function definition at end of file.

### 2026-02-08 — P1/P2 Fixes (session 3)

7. **`_save_graph` no-op + duplicate save block** (`pipeline.py`) —
   - Wired `_save_graph` body to call `export_graph_to_gfa(graph, output_path)`
     (already imported from `io_utils.assembly_export`).  Added try/except with
     warning-level log on export failure.
   - Removed duplicate save block in `_step_assemble` (lines 601–614 were a
     near-identical copy of lines 587–598).  The second block only saved
     `assembly_result.dbg`, overwriting the first block's
     `string_graph or dbg` selection and losing the string graph.

8. **Checkpoint resume doesn't persist pipeline state** (`pipeline.py`) —
   - `_create_checkpoint` now serialises a `pipeline_state` dict containing
     `read_files`, `technologies`, `kmer_prediction`, and `corrected_files`
     alongside the existing step/timestamp metadata.
   - `_find_last_checkpoint` now restores those state keys (plus
     `completed_steps`) into `self.state` before returning the resume step.

9. **`_read_file_streaming` doesn't handle `.gz`** (`pipeline.py`) —
   - Now strips `.gz`/`.gzip` suffix before checking the real format extension
     (`.fq`, `.fastq`, `.fa`, `.fasta`).  The actual `read_fastq` / `read_fasta`
     callees already handle gzip decompression natively via `gzip.open`.
   - Replaced bare `except:` with `except Exception:` in the fallback branch.

### 2026-02-08 — P2 Nextflow Pipeline Fix (session 3, continued)

10. **Nextflow pipeline non-functional — 9 missing CLI subcommands** (multi-file) —
    The Nextflow workflow invoked 9 `strandweaver` subcommands that did not exist
    in the CLI.  Added all 9 as functional top-level commands in `cli.py`:
    `classify`, `kweaver`, `nf-build-graph`, `nf-edgewarden-filter`,
    `nf-pathweaver-iter-general`, `nf-threadcompass-aggregate`, `nf-strandtether-phase`,
    `nf-pathweaver-iter-strict`, `nf-export-assembly`.  Each command wires to the
    corresponding internal Python module (read classification, KWeaverPredictor,
    DeBruijnGraphBuilder, EdgeWarden, PathWeaver, ThreadCompass, StrandTether,
    and assembly_export respectively).

    **Supporting changes:**
    - **GFA reader** (`io_utils/assembly_export.py`): Added `load_graph_from_gfa()`
      to reconstruct a `KmerGraph` from GFA v1 S-lines and L-lines.  The
      Nextflow pipeline passes GFA files between steps, so every step command
      needs to load graphs.  Exported via `io_utils/__init__.py`.
    - **Batch `extract-kmers` arg mismatch** (`cli.py`): Nextflow passes
      `--reads` and `--kmer-predictions` but CLI only accepted `--input` and
      `--kmer-size`.  Added `--reads` alias and `--kmer-predictions` option;
      k is now resolved from predictions JSON when available.
    - **Batch `merge-svs` arg mismatch** (`cli.py`): Nextflow passes `--vcfs`
      and `--summary` but CLI only accepted `--input`.  Added `--vcfs` alias
      and `--summary` option.
    - **Channel truthiness bug** (`strandweaver.nf`): `if (ont_ul_reads)` and
      `if (hic_r1 && hic_r2)` were always truthy (channels are DSL2 objects).
      Changed to `if (params.ont_ul)` and `if (params.hic_r1 && params.hic_r2)`.
    - **Helper function stubs** (`strandweaver.nf`): `extractEdges()` and
      `partitionGraph()` returned empty lists.  Implemented both to parse GFA
      S/L lines and split into batch files.
    - **Inconsistent invocation** (8 `.nf` modules): Standardised from
      `python3 -m strandweaver.cli` to `strandweaver` entry point across all
      modules.

### 2026-02-08 — P2 AI Model Loading Wiring (session 4)

11. **`_load_ai_models()` set all 7 model slots to `None`** (multi-file) —
    Every model slot (`adaptive_kmer`, `base_error_classifier`, `edge_ai`,
    `path_gnn`, `diploid_ai`, `ul_routing_ai`, `sv_ai`) was unconditionally
    `None` with a `# TODO: Load actual model` comment.  The pipeline ran in
    full classical-fallback mode even when `--use-ai` was set (the default).

    **Fixes applied:**
    - **`_load_ai_models()` rewrite** (`pipeline.py`): Each model slot now
      resolves its checkpoint from a three-level priority chain:
      per-model `model_path` in config → central `model_dir/sub_folder` →
      package default location.  Pickle models are loaded directly;
      EdgeWarden and PathGNN store resolved directory/file paths (the class
      constructors handle their own loading).  Clear warnings are emitted for
      every model that cannot be located, and a summary line reports
      loaded-vs-fallback counts.
    - **Model resolution helpers** (`pipeline.py`): Added `_resolve_model_dir()`,
      `_resolve_sub_model_path()`, `_try_load_pickle_model()`, and
      `_try_load_pickle_file()` private methods to keep loading logic DRY.
    - **`save_all_models()` method** (`pipeline.py`): New public method that
      exports all currently-loaded models to the canonical directory layout
      expected by `_load_ai_models()`.  Intended for use after training.
    - **EdgeWarden wiring** (`pipeline.py`): All 3 `EdgeWarden(technology=…)`
      instantiation sites now pass `model_dir=` from the loaded AI config and
      call `load_models()` so pre-trained per-technology pkl files are loaded
      automatically.
    - **KWeaverPredictor wiring** (`pipeline.py`): `_step_kweaver()` now
      resolves and passes `model_dir=` to `KWeaverPredictor(…)` so trained
      XGBoost models are found when present.
    - **Config schema** (`config/schema.py`): Added top-level
      `ai.model_dir: None` key with layout documentation comment.
    - **CLI** (`cli.py`): Added `--model-dir` option to `pipeline` command,
      wired to `pipeline_config['ai']['model_dir']`.

### 2026-02-09 — Remaining findings clean-up (session 5)

12. **Paired-end pairing lost during correction** (`cli.py`, `pipeline.py`) —
    `--illumina-r1` / `--illumina-r2` appended both files into the flat
    `all_reads` list with technologies `['illumina', 'illumina']`.  In
    `_step_correct`, every read was output as `corrected_illumina_N.fastq`
    with no R1/R2 label, making downstream paired-end alignment impossible.

    **Fixes:**
    - **CLI** (`cli.py`): When `--illumina-r1` and `--illumina-r2` are both
      set, records `illumina_paired_indices = (len(all_reads), len(all_reads)+1)`
      *before* extending `all_reads`.  Stored in
      `pipeline_config['runtime']['illumina_paired_indices']`.
    - **`_step_correct`** (`pipeline.py`): Looks up `paired_idx` set from
      config.  When the current file index is in `paired_idx`, appends `_R1`
      or `_R2` to the output filename.  After correction, stores a
      `self.state['illumina_paired_corrected']` dict with `r1`/`r2` paths
      for downstream consumption.

13. **`hic_contact_map` and `graph` never stored in pipeline state**
    (`pipeline.py`) —
    `_step_classify_chromosomes` reads `self.state.get('graph')` and
    `self.state.get('hic_contact_map')` but neither was ever populated.

    **Fixes:**
    - **`_add_hic_edges_to_graph`**: After
      `contact_map = tether.build_contact_map(hic_pairs)`, added
      `self.state['hic_contact_map'] = contact_map`.
    - **`_step_assemble`**: After assembly, added
      `self.state['graph'] = assembly_result.string_graph or assembly_result.dbg`.

14. **`assemble()` stub in `assembly_core/__init__.py`** —
    The public `assemble()` function raised `NotImplementedError`.  A dead
    `AssemblyGraph` class was also present.

    **Fix:** Replaced stub with a real implementation that reads input
    FASTA/FASTQ via `read_fasta`/`read_fastq`, builds a de Bruijn graph
    via `build_dbg_from_long_reads`, extracts contigs from graph nodes
    ≥ `min_contig_length`, writes output FASTA, and returns a stats dict.
    Removed the dead `AssemblyGraph` class.
