# StrandWeaver Genomics Audit — February 2026

Systematic review of pipeline logic, scoring models, and biological assumptions.

## Status Key
- 🔴 CRITICAL — Would produce wrong assemblies
- 🟠 HIGH — Significant accuracy loss
- 🟡 MODERATE — Suboptimal but not catastrophic
- ✅ FIXED — Resolved with commit reference

---

## 🔴 CRITICAL

| ID | Module | Issue | Reference |
|----|--------|-------|-----------|
| G1 | StrandTether | ✅ **FIXED** — Hi-C orientation scoring was inverted. `++`/`--` scored 1.0, `+-`/`-+` scored 0.0. Valid Hi-C pairs are convergent (`+-`/`-+`). | Lieberman-Aiden et al. *Science* 2009; SALSA2 |
| G2 | HaplotypeDetangler / StrandTether | ✅ **FIXED** — Replaced global spectral clustering with bubble-aware local phasing. Detects heterozygous bubbles in the assembly graph, bins Hi-C contacts per-bubble, chains local phase decisions via BFS. Added `ploidy` parameter (haploid assemblies skip phasing). Old spectral method kept as fallback when no bubbles detected. | hifiasm (Cheng et al. *Nat Methods* 2021) |
| G3 | Pipeline | ✅ **FIXED** — String graph contig extraction was a placeholder. Now traverses combined DBG + UL overlay edges. | pipeline.py `_extract_contigs_from_string_graph()` |
| G4 | Pipeline | ✅ **FIXED** — Hi-C files now excluded from primary tech vote. | pipeline.py `_step_assemble()` |
| G5 | Pipeline | ✅ **FIXED** — K-Weaver now selects primary assembly file (HiFi > ONT > Illumina), skipping Hi-C/UL. | pipeline.py `_step_kweaver()` |

## 🟠 HIGH

| ID | Module | Issue | Reference |
|----|--------|-------|-----------|
| G6 | StrandTether | ✅ **FIXED** — Added `calibrate_contacts()` with ICE-like iterative marginal balancing (3 iterations) to correct GC/mappability/fragment biases. | Imakaev et al. *Nat Methods* 2012 |
| G7 | StrandTether | ✅ **FIXED** — `score_join()` now accepts `genomic_distance` and applies C(s) ∝ s^{-α} decay correction using `distance_decay_power`. | Hi-C: C(s) ∝ s^-α, α ≈ 1.08 |
| G8 | StrandTether | ✅ **FIXED** — Contact normalization now uses log-scaling with calibrated 95th-percentile ceiling. Handles 10²–10⁴ range. | Rao et al. *Cell* 2014 |
| G9 | StrandTether | ✅ **FIXED** — Join scoring weights now computed as `1.0 - orient_weight - distance_weight` for contact term. | Arithmetic error |
| G10 | StrandTether | ✅ **FIXED** — Label propagation now seeds deterministically: sorted by total contacts (desc), breaking ties by node ID. | hifiasm seeds from het k-mer pairs |
| G11 | Pipeline | ✅ **FIXED** — Error profiler k clamped to max 21 regardless of K-Weaver DBG k. | pipeline.py `_step_profile()` |
| G12 | EdgeWarden | ✅ **FIXED** — Mismatch thresholds now technology-specific: HiFi 1%, ONT R9 8%, R10 5%, Illumina 1%, aDNA 15%. | Wenger et al. 2019; Wick et al. 2023 |
| G13 | EdgeWarden | ✅ **FIXED** — Coverage ratio threshold lowered from 5.0×/4.0× to 2.5× in both CascadeClassifier and HybridEnsemble. | Merqury (Rhie et al. 2020) |
| G14 | PathWeaver | ✅ **FIXED** — SV penalty now severity-based: <0.3 → hard cap at 0.10, <0.5 → 50% reduction, <0.8 → up to 25% reduction. | T2T-CHM13 (Rhie et al. *Nature* 2021) |

## 🟡 MODERATE

| ID | Module | Issue | Reference |
|----|--------|-------|-----------|
| G15 | DBG Engine | ✅ **FIXED** — `max_error_differences` now technology-aware: `max(2, ceil(error_rate × arm_length × 3))`. ONT bubbles with 3+ diffs in homozygous arms no longer incorrectly protected. | Scale with error rate + arm length |
| G16 | DBG Engine | ~~Tip length uses `k-2` overlap instead of `k-1`.~~ **NOT A BUG** — nodes are (k-1)-mers, so k-2 overlap is correct. | Verified in dbg_engine_module.py L702-706 |
| G17 | Pipeline | ✅ **FIXED** — ONT default k lowered from 41 to 21 without K-Weaver. P(correct 41-mer) at 10% error ≈ 1.3%. | K-Weaver correctly predicts k=21 |
| G18 | Pipeline | ✅ **FIXED** — HiFi default k lowered from 51 to 31 to match K-Weaver rules. k=51 drops connectivity at <30× coverage. | hifiasm uses k=51 after error correction |
| G19 | K-Weaver | ✅ **FIXED** — Genome size estimation now uses k-mer histogram peak method instead of unique k-mer count. Handles repetitive genomes correctly. | GenomeScope-style peak detection |
| G20 | ErrorSmith | ✅ **FIXED** — `PacBioCorrector.build_global_spectrum()` builds population k-mer table from all reads before per-read correction. Pipeline wired to call it automatically. | hifiasm builds global tables first |
| G21 | ErrorSmith | ✅ **FIXED** — Ancient DNA damage rate default raised from 5% to 30%. Added `estimate_damage_from_reads()` for data-driven estimation from C→T/G→A frequencies. | Briggs et al. *PNAS* 2007 |
| G22 | PathWeaver | ✅ **FIXED** — Hi-C weight raised from 15% to 25% (dominant signal). Weights rebalanced to sum to 1.0: EdW=20%, GNN=20%, Topo=15%, UL=15%, HiC=25%, Val=5%. | 3D-DNA, SALSA2, YaHS |
| G23 | DBG Engine | ✅ **FIXED** — Even k-mers now raise `ValueError` instead of warning. Palindrome ambiguity in canonical DBG is a correctness issue. | Standard convention: odd k only |
| G24 | K-Weaver | ✅ **FIXED** — `ul_overlap_k` capped from 1001 to 501 (value was advisory; UL mapper uses `anchor_k=15`). Docstring clarified. | Dead prediction cleaned up |
