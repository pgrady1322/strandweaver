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
| G2 | StrandTether | Spectral clustering (Fiedler vector) on full contact graph separates chromosomes, not haplotypes. Diploid phasing requires bubble-aware local phasing. | hifiasm (Cheng et al. *Nat Methods* 2021) |
| G3 | Pipeline | ✅ **FIXED** — String graph contig extraction was a placeholder. Now traverses combined DBG + UL overlay edges. | pipeline.py `_extract_contigs_from_string_graph()` |
| G4 | Pipeline | ✅ **FIXED** — Hi-C files now excluded from primary tech vote. | pipeline.py `_step_assemble()` |
| G5 | Pipeline | ✅ **FIXED** — K-Weaver now selects primary assembly file (HiFi > ONT > Illumina), skipping Hi-C/UL. | pipeline.py `_step_kweaver()` |

## 🟠 HIGH

| ID | Module | Issue | Reference |
|----|--------|-------|-----------|
| G6 | StrandTether | No ICE/KR matrix balancing on Hi-C contacts. GC, mappability, fragment-length biases uncorrected. | Imakaev et al. *Nat Methods* 2012 |
| G7 | StrandTether | `distance_decay_power=1.5` declared but never used. No genomic-distance correction. | Hi-C: C(s) ∝ s^-α, α ≈ 1.08 |
| G8 | StrandTether | `max_expected = threshold × 20` saturates at ~40 contacts. Real data has 10²–10⁴. All variation lost. | Rao et al. *Cell* 2014 |
| G9 | StrandTether | ✅ **FIXED** — Join scoring weights now computed as `1.0 - orient_weight - distance_weight` for contact term. | Arithmetic error |
| G10 | StrandTether | Label propagation seeds arbitrary — first two nodes in set iteration order. Non-reproducible phasing. | hifiasm seeds from het k-mer pairs |
| G11 | Pipeline | Error profiler uses DBG k (up to 51) instead of small k (17–21). No frequency signal at high k. | pipeline.py `_step_profile()` |
| G12 | EdgeWarden | Mismatch threshold 10% is technology-agnostic. HiFi should be ~0.1%; ONT R9 ~5–10%. | Wenger et al. 2019; Wick et al. 2023 |
| G13 | EdgeWarden | Coverage ratio threshold 5.0× for repeat detection. Diploid collapse starts at 2×. Should be ~2.0–2.5×. | Merqury (Rhie et al. 2020) |
| G14 | PathWeaver | SV misassembly penalty capped at 10% score reduction. Confirmed misassembly should be disqualifying. | T2T-CHM13 (Rhie et al. *Nature* 2021) |

## 🟡 MODERATE

| ID | Module | Issue | Reference |
|----|--------|-------|-----------|
| G15 | DBG Engine | Bubble popping `max_error_differences=2` is technology-agnostic. ONT errors can produce 3+ diffs in homozygous bubbles. | Scale with error rate + arm length |
| G16 | DBG Engine | ~~Tip length uses `k-2` overlap instead of `k-1`.~~ **NOT A BUG** — nodes are (k-1)-mers, so k-2 overlap is correct. | Verified in dbg_engine_module.py L702-706 |
| G17 | Pipeline | ONT default k=41 without K-Weaver. P(correct 41-mer) at 10% error ≈ 1.3%. | K-Weaver correctly predicts k=21 |
| G18 | Pipeline | HiFi default k=51 disagrees with K-Weaver rules (k=31). k=51 drops connectivity at <30× coverage. | hifiasm uses k=51 after error correction |
| G19 | K-Weaver | Genome size from 10k read sample underestimates by ~1000× for repetitive genomes. | Inflates coverage in ML features |
| G20 | ErrorSmith | HiFi per-read k-mer correction without global k-mer table. No population frequency signal. | hifiasm builds global tables first |
| G21 | ErrorSmith | Ancient DNA damage rate hardcoded at 5%. Degraded specimens (>10k yrs) show 30–40%. | Briggs et al. *PNAS* 2007 |
| G22 | PathWeaver | Hi-C gets only 15% weight in final path scoring. Published scaffolders use Hi-C as dominant signal. | 3D-DNA, SALSA2, YaHS |
| G23 | DBG Engine | Even k-mers warned but allowed. Creates palindrome ambiguity in canonical DBG. | Standard convention: odd k only |
| G24 | K-Weaver | `extension_k=1001` for UL reads never consumed downstream. UL mapper uses `anchor_k=15`. | Dead prediction |
