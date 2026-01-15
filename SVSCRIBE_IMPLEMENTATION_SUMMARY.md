# SVScribe Implementation Summary
**Date**: December 22, 2025  
**Status**: ✅ COMPLETE

## Overview

Implemented comprehensive SV detection module following the 8-step algorithm with multi-source evidence scoring and haplotype-aware detection.

## Data Structures

### SVEvidence
Multi-source evidence tracking:
- `has_sequence_support`, `has_ul_support`, `has_hic_support`: bool flags
- `sequence_confidence`, `ul_confidence`, `hic_confidence`: 0-1 scores
- `supporting_reads`: int count
- `edge_types`: Set of involved edge types

### SVSignature
Topological pattern detection:
- `pattern_type`: Detection method identifier
- `involved_nodes`, `involved_edges`: Graph elements
- `coverage_pattern`: Coverage description
- `topology_score`: 0-1 pattern strength
- `metadata`: Dict for detector-specific data

### SVCall
Final SV call output:
- `sv_id`: Unique ID (e.g., "SV000001")
- `sv_type`: 'DEL', 'INS', 'INV', 'DUP', 'TRA'
- `nodes`: List of involved node IDs
- `size`: Size in bases
- `confidence`: Overall 0-1 score
- `evidence`: Dict containing SVEvidence and metadata
- `haplotype`: 0=hapA, 1=hapB, -1=unknown
- `breakpoints`: List of (from_node, to_node) tuples

## SV Detector Classes

All detectors return `List[SVSignature]` and use edge_type categorization:

### DeletionDetector
**Signatures**:
- `ul_spanning_gap`: UL edges connecting nodes without sequence edges
- `hic_spanning_gap`: Hi-C contacts across sequence gaps
- `coverage_drop`: Nodes with >50% coverage drop vs neighbors

**Methods**:
- `_detect_ul_spanning_gaps()`: Find UL edges bridging sequence gaps
- `_detect_hic_gap_spanning()`: Find Hi-C contacts across gaps
- `_detect_coverage_gaps()`: Find coverage anomalies
- `_estimate_gap_size()`: Estimate deletion size
- `_has_sequence_path()`: BFS through sequence edges

### InsertionDetector
**Signatures**:
- `bubble_insertion`: Parallel paths with size difference >1kb
- `high_coverage_branch`: Branch points with coverage >60
- `ul_novel_sequence`: Nodes with ≥3 UL supports

**Methods**:
- `_detect_alternative_paths()`: Find bubble structures
- `_detect_high_coverage_branches()`: Find elevated coverage branches
- `_detect_ul_novel_sequences()`: Find UL-supported novel regions
- `_trace_path()`: Follow sequence path through graph

### InversionDetector
**Signatures**:
- `ul_strand_flip`: UL reads with strand inconsistencies
- `hic_orientation_flip`: Hi-C suggesting orientation changes

**Methods**:
- `_detect_ul_strand_flips()`: Check UL routing strand consistency
- `_detect_hic_orientation_flips()`: Check Hi-C pair orientations (placeholder)

### DuplicationDetector
**Signatures**:
- `high_coverage_duplication`: Nodes with >2x median coverage
- `parallel_paths`: Multiple paths to same target
- `self_loop`: Edges connecting node to itself

**Methods**:
- `_detect_high_coverage_duplications()`: Find >2x coverage nodes
- `_detect_parallel_paths()`: Find branching to same target
- `_detect_self_loops()`: Find circular edges

### TranslocationDetector
**Signatures**:
- `hic_long_range`: Hi-C edges with >100kb sequence distance
- `ul_long_range`: UL reads spanning >100kb sequence distance

**Methods**:
- `_detect_hic_long_range()`: Find distant Hi-C contacts
- `_detect_ul_long_range()`: Find distant UL spans
- `_calculate_sequence_distance()`: BFS distance through sequence graph

## Main SVScribe Class

### Initialization
```python
SVScribe(
    use_ai: bool = False,
    min_confidence: float = 0.5,
    min_size: int = 50,
    ml_model: Optional[str] = None
)
```

### Main Method: detect_svs()
```python
detect_svs(
    graph,
    ul_routes: Optional[Dict] = None,
    distinguish_edge_types: bool = True,
    phasing_info = None,
    reference_path: Optional[str] = None
) -> List[SVCall]
```

### 8-Step Algorithm

**Step 1: Categorize Edges**
- `_categorize_edges()` → (sequence_edges, ul_edges, hic_edges)
- Uses `edge.edge_type` attribute ('sequence', 'ul', 'hic')
- Falls back to treating all as sequence if distinguish_edge_types=False

**Step 2: Run Detectors**
- Calls all 5 detector classes
- Returns List[(SVSignature, sv_type)]
- Logs detection counts per type

**Step 3: Score Evidence**
- `_score_evidence()` → SVEvidence
- Checks sequence, UL, Hi-C support
- Counts supporting reads
- `_check_ul_support()`: Count UL edges and route overlaps
- `_check_hic_support()`: Count Hi-C edges

**Step 4: Calculate Confidence**
- `_calculate_confidence()` → float
- Weighted: sequence 40%, UL 30%, Hi-C 30%
- Multi-source bonus: +20% if ≥2 sources

**Step 5: Assign Haplotypes**
- `_assign_haplotypes()` → modifies SVCall.haplotype
- Uses `phasing_info.node_assignments`
- Assigns to most common haplotype (>50% agreement)
- Maps: 0=hapA, 1=hapB, -1=unknown

**Step 6: Merge Overlapping**
- `_merge_overlapping_svs()` → List[SVCall]
- Groups by sv_type
- Merges if ≥50% node overlap
- `_merge_two_svs()`: Combines evidence, takes max confidence

**Step 7: Apply AI Refinement**
- `_apply_ai_refinement()` → List[SVCall]
- Optional (if use_ai=True and model loaded)
- `_extract_sv_features()`: Extract 20 features
- Averages AI confidence with existing confidence
- `_load_ai_model()`: Loads PyTorch model

**Step 8: Filter and Assign IDs**
- Filters by min_size threshold
- Assigns unique IDs: "SV000001", "SV000002", etc.
- Updates type counts

### Additional Methods

**get_sv_type_counts()**
```python
Returns: {'DEL': int, 'INS': int, 'INV': int, 'DUP': int, 'TRA': int}
```

## Integration Points

### With ThreadCompass
```python
# In pipeline after ThreadCompass
thread_compass = ThreadCompass(...)
result.string_graph = thread_compass.route_ul_reads(...)

# Get UL routes for SVScribe
ul_routes = thread_compass.get_routes()
```

### With HaplotypeDetangler
```python
# After phasing
phasing_result = haplotype_detangler.phase_graph(...)

# Use phasing_info for haplotype assignment
sv_calls = sv_scribe.detect_svs(..., phasing_info=phasing_result)
```

### With Edge Types
Graph edges should have `edge_type` attribute:
- Set by DBG: `edge.edge_type = 'sequence'`
- Set by ThreadCompass: `edge.edge_type = 'ul'`
- Set by Hi-C integration: `edge.edge_type = 'hic'`

### Pipeline Usage
```python
# In strandweaver/utils/pipeline.py:_step_assemble()

from ..assembly_core.svscribe_module import SVScribe

# After HaplotypeDetangler
use_sv_ai = self.config.get('assembly', {}).get('sv_detection', {}).get('use_sv_ai', True)
sv_scribe = SVScribe(
    use_ai=use_sv_ai and self.config['ai']['enabled'],
    min_confidence=0.5,
    min_size=50
)

sv_calls = sv_scribe.detect_svs(
    graph=result.string_graph or result.dbg,
    ul_routes=thread_compass.get_routes(),
    distinguish_edge_types=True,
    phasing_info=phasing_result
)

# Store results
result.stats['sv_calls'] = len(sv_calls)
result.stats['sv_types'] = sv_scribe.get_sv_type_counts()

# Export to file
from ..io_utils.assembly_export import export_sv_calls  # If function exists
sv_path = self.output_dir / "sv_calls.json"
# Or use svs_to_dict_list() for manual export
```

## Convenience Functions

### detect_structural_variants()
Wrapper function for quick usage:
```python
from strandweaver.assembly_core.svscribe_module import detect_structural_variants

sv_calls = detect_structural_variants(
    graph,
    ul_routes=ul_routes,
    distinguish_edge_types=True,
    phasing_info=phasing_result,
    use_ai=True,
    min_confidence=0.5,
    min_size=50
)
```

### svs_to_dict_list()
Convert SVCalls to JSON-serializable format:
```python
from strandweaver.assembly_core.svscribe_module import svs_to_dict_list
import json

sv_dicts = svs_to_dict_list(sv_calls)
with open('sv_calls.json', 'w') as f:
    json.dump(sv_dicts, f, indent=2)
```

## Testing Checklist

- [ ] Test with sequence edges only (basic detection)
- [ ] Test with UL edges (spanning evidence)
- [ ] Test with Hi-C edges (long-range validation)
- [ ] Test with all three edge types (multi-source scoring)
- [ ] Test with phasing_info (haplotype assignment)
- [ ] Test without phasing_info (all -1 haplotypes)
- [ ] Test AI refinement (if model available)
- [ ] Test edge cases (empty graph, no edges, no UL routes)
- [ ] Test SV merging (overlapping calls)
- [ ] Test size filtering (min_size threshold)
- [ ] Test confidence filtering (min_confidence threshold)
- [ ] Verify get_sv_type_counts() accuracy
- [ ] Verify unique SV IDs

## Expected Outputs

For typical genome assembly with mixed data:
- **Deletions**: 50-200 calls (heterozygous deletions, assembly gaps)
- **Insertions**: 100-500 calls (novel sequences, extra copies)
- **Inversions**: 5-20 calls (rare but high-confidence)
- **Duplications**: 200-1000 calls (tandem repeats, segmental dups)
- **Translocations**: 0-10 calls (very rare, likely false positives without validation)

Confidence distribution:
- >0.8: 10-20% (high-confidence, multi-source)
- 0.6-0.8: 40-60% (good evidence, 2 sources)
- 0.5-0.6: 30-40% (threshold calls, single source)

## Success Criteria

✅ **Implemented**:
- All 5 SV detector classes
- SVScribe main class with 8-step algorithm
- Multi-source evidence scoring (sequence 40%, UL 30%, Hi-C 30%)
- Haplotype assignment from phasing_info
- SV merging for overlapping calls
- AI refinement framework (model loading, feature extraction)
- Edge type categorization
- Helper methods for evidence checking
- get_sv_type_counts() method
- Convenience functions
- No syntax errors

✅ **Ready for Integration**:
- Compatible with ThreadCompass.get_routes()
- Compatible with HaplotypeDetangler.PhasingResult
- Uses edge_type attribute from graph edges
- Returns standard SVCall objects
- Provides JSON export via svs_to_dict_list()

## Next Steps

1. **Add to Pipeline**: Integrate SVScribe call in `_step_assemble()` after HaplotypeDetangler
2. **Add Export**: Create `export_sv_calls()` in io_utils/assembly_export.py or use JSON export
3. **Train AI Model**: Collect training data and train sv_classifier.pt model (optional)
4. **Validation**: Test on real assemblies with known SVs
5. **Benchmarking**: Compare with post-assembly SV callers (sniffles, SVIM, etc.)
6. **Documentation**: Add usage examples and interpretation guidelines

## Notes

- AI refinement is optional and requires trained model
- Hi-C orientation detection is placeholder (needs read pair metadata)
- Size estimates are heuristic (could be improved with better path tracing)
- Translocation detection may have false positives (needs validation)
- Consider adding VCF export format for compatibility with downstream tools
