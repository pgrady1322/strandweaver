# ThreadCompass Improvement Plan (Tier 2+)

Date: Dec 16, 2025
Status: Draft (aligned with PathWeaver updates)

## Goals
- Integrate long-range evidence (Hi-C, UL) for path validation and threading
- Apply advanced validation rules where long-range data is available
- Provide confidence-aware stitching between contigs with robust fallbacks

## Planned Enhancements

- Long-Range Support Rules (deferred until module input stage):
  - Hi-C contact consistency checks at joins (intra-chromosomal vs inter-chromosomal likelihoods)
  - UL read span validation across suspected misjoins (minimum spanning count and identity)
  - Orientation and distance constraints from long-range signals

- Confidence-Aware Stitching:
  - Thresholded break/rejoin based on combined EdgeWarden + GNN + long-range support
  - Safe merges only when support exceeds configurable minimums
  - Record per-join uncertainty and rationale for downstream QC

- Ambiguity Handling:
  - Maintain alternative thread candidates with scores when evidence is conflicting
  - Export candidates for manual curation when uncertainty is high

## Data Inputs
- Hi-C matrices / contact maps (normalized)
- UL read mappings (chain alignments)
- EdgeWarden edge metrics (confidence, repeat, coverage consistency)
- PathWeaver path metadata (misassembly flags, uncertainty)

## API Sketch
```python
class ThreadCompass:
    def validate_join(self, left_path, right_path, hic_matrix=None, ul_reads=None, thresholds=None):
        # Combine EdgeWarden + GNN + long-range support to accept/reject
        pass
```

## Milestones
- M1: Define data adapters for Hi-C and UL inputs
- M2: Implement validation rules and thresholds
- M3: Integrate with PathWeaver outputs and misassembly flags
- M4: Tests + benchmarking on known datasets

## Notes
- These rules are deferred here; PathWeaver excludes Hi-C/UL at this stage
- Once ThreadCompass ingests long-range data, enable the above checks
