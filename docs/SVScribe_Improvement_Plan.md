# SVScribe Improvement Plan (Tier 2+)

Date: Dec 16, 2025
Status: Draft (aligned with PathWeaver updates)

## Goals
- Structural variant calling informed by long-range evidence (Hi-C, UL)
- Confidence-aware breakpoint validation and classification
- Export rich QC metadata for downstream review

## Planned Enhancements

- Long-Range Breakpoint Validation (deferred until module input stage):
  - Hi-C support: breakpoint contact anomalies and cis/trans evidence
  - UL support: read-span and split-read confirmation across breakpoints
  - Orientation and distance consistency checks

- Confidence and Uncertainty Modeling:
  - Fuse EdgeWarden + GNN + long-range signals into breakpoint score
  - Track uncertainty and evidence breakdown per call
  - Thresholds for high/medium/low confidence SVs

- Reporting & QC:
  - Emit TSV/JSON with evidence counts, contact scores, uncertainty
  - Flag candidates needing manual review

## Data Inputs
- Hi-C contact maps
- UL read chains
- EdgeWarden edge metrics
- PathWeaver misassembly flags and path metadata

## API Sketch
```python
class SVScribe:
    def validate_breakpoint(self, breakpoint, hic_matrix=None, ul_reads=None, thresholds=None):
        # Combine signals to confirm/score SVs
        pass
```

## Milestones
- M1: Define data adapters for Hi-C and UL inputs
- M2: Implement long-range breakpoint validation rules
- M3: Integrate with PathWeaver/ThreadCompass outputs
- M4: Tests + benchmarking against curated SV sets

## Notes
- Deferred here; PathWeaver excludes Hi-C/UL until downstream modules
- Enable these enhancements in SVScribe when long-range data becomes available
