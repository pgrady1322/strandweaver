# StrandWeaver — Future Enhancements

Tracked issues, planned features, and integration gaps identified during development.

---

## Priority: HIGH

### 1. Per-Technology Error Profiling in Pipeline

**Status:** ✅ Implemented  
**Affected:** `_step_profile()` in `pipeline.py`, `_step_correct()` in `pipeline.py`

The pipeline's profiling step now profiles **every** `(read_file, technology)` pair independently. Each sequencing technology has fundamentally different error profiles (substitution-dominated for Illumina, indel-dominated for ONT, low-error for HiFi).

**Implementation:**
- `_step_profile()` loops over all read files and generates a separate `error_profile_{tech}_{i}.json` for each
- Profiles are stored in `self.state['error_profiles']` (dict keyed by technology string)
- `_step_correct()` looks up `error_profiles.get(tech)` and passes the matching profile to the corrector
- Hi-C reads are automatically skipped (proximity ligation, not error-corrected)
- Backward compatibility: `self.state['error_profile']` still holds the first profile

---

### 2. PacBio Flowcell / Chemistry Support in Profile

**Status:** Not implemented  
**Affected:** `profile` CLI command

ONT reads support `--ont-flowcell`, `--ont-basecaller`, `--ont-accuracy` metadata flags for technology-aware profiling. PacBio HiFi reads have analogous metadata (Sequel II vs Revio, CCS version, chemistry version) that could improve profiling accuracy, but no equivalent flags exist.

**Proposed additions:**
- `--pacbio-platform` — Platform type (`sequel2`, `revio`)
- `--pacbio-chemistry` — Chemistry version (e.g., `2.0`, `3.0`)
- `--pacbio-ccs-version` — CCS software version for kinetics-aware profiling
- Parse PacBio BAM header (`@RG` tags) for automatic detection

---

## Priority: MEDIUM

### 3. LongBow ONT Auto-Detection (`--ont-detect`)

**Status:** Wired but placeholder  
**Affected:** `profile` CLI command, `read_classification_utility.py`

The `--ont-detect` flag is connected to `detect_nanopore_metadata_with_longbow()`, which calls `run_longbow()`. However, the LongBow command interface is **placeholder code** — the actual CLI arguments have not been verified against the real LongBow tool:

```python
# Note: Adjust command based on actual LongBow CLI
# This is a placeholder - need to verify actual LongBow interface
cmd = ['longbow', 'detect', '--input', str(reads_file), ...]
```

**Background:** LongBow (PacBio / Broadband toolkit) is designed for PacBio MAS-Seq/Kinnex demultiplexing and array element segmentation. Its relevance to ONT metadata detection is limited. The original intent was likely to auto-detect basecaller, flowcell, and chemistry metadata from read headers to inform technology-specific profiling.

**Action needed:**
1. **Verify LongBow scope** — LongBow is PacBio-focused (MAS-Seq adapter identification). It is unlikely to be the right tool for ONT metadata extraction. Confirm and remove if inappropriate.
2. **Native ONT metadata detection** — Implement direct parsing of:
   - **POD5 headers**: `pod5 inspect reads <file>` provides run_id, flow_cell_id, experiment_id, basecall_model, etc.
   - **FAST5 files**: HDF5 `context_tags` and `tracking_id` groups contain flowcell type, kit, basecaller version.
   - **FASTQ headers**: Dorado and Guppy encode basecaller model in `@` lines and `RG` tags for BAM.
3. **Dorado `summary` integration** — `dorado summary <bam>` produces a TSV with flowcell_id, kit, basecalling_model, mean_qscore per read — lightweight and already installed in many ONT environments.
4. **Fallback chain**: Try POD5 headers → dorado summary → FAST5 HDF5 → FASTQ header regex → user-provided `--ont-flowcell` / `--ont-basecaller` flags.

**Estimated effort:** Medium — the POD5 Python API (`pod5.Reader`) and `h5py` for FAST5 are well-documented; FASTQ regex parsing is trivial.

---

### 4. Standalone `assemble` and `build-contigs` Commands

**Status:** Stub only  
**Affected:** `assemble` CLI command, `build-contigs` CLI command

Both commands print "This feature is planned for v0.2" and do not execute any assembly logic. The full pipeline (`pipeline` command) handles assembly internally through `_step_assemble()`. These standalone commands should either:

1. Be wired to the same assembly engine used by the pipeline, or
2. Be removed if standalone assembly is not a supported use case

---

### 5. Multi-Technology Read Merging (`merge` command)

**Status:** Stub only  
**Affected:** `merge` CLI command

The `merge` command prints "This feature is planned for v0.2" without executing. The pipeline handles multi-technology reads natively, but a standalone merge utility could be useful for users working with external assemblers.

---

## Priority: LOW

### 6. Polishing and Gap Filling in Finish Step

**Status:** TODO stubs  
**Affected:** `_step_finish()` in `pipeline.py`

The finish step has placeholder stubs for polishing and gap filling:

```python
# TODO: Implement polishing
# TODO: Implement gap filling
```

These are partially addressed by the T2T-Polish integration referenced in the pipeline display, but the in-pipeline implementations are not connected.

---

### 7. Validate Command — Reference Comparison

**Status:** Stub only  
**Affected:** `validate` CLI command

The `validate` command accepts a `--reference` flag for reference-based validation but the comparison logic is not implemented. Basic N50/L50 statistics work.

---

## Removed Features (Cleaned Up)

| Feature | Reason | Date |
|---------|--------|------|
| `-r` / `--technology` old syntax | Never released; replaced by `-r1`/`--technology1` numbered syntax | 2026-02-09 |
| `--ai-finish` / `--no-ai-finish` | Idea never developed; finishing handled by pipeline modules | 2026-02-09 |
| `finish` standalone CLI command | Dead code — printed stub message only; `_step_finish()` retained in pipeline | 2026-02-09 |
| Claude API config in `schema.py` | Associated with removed `--ai-finish`; no implementation existed | 2026-02-09 |
