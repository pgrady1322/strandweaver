# Coordinator Consolidation Summary

**Date**: December 18, 2024  
**Consolidation**: Three separate coordinators merged into one master orchestrator

---

## Overview

Successfully consolidated three separate coordinator classes into a single unified **`PipelineOrchestrator`** that manages the complete StrandWeaver assembly pipeline from start to finish.

---

## Files Consolidated

### 1. **PreprocessingCoordinator** (403 lines)
- **Former locations**: 
  - `assembly_utils/utilities.py` 
  - `assembly/utilities.py` (duplicate)
- **Purpose**: K-Weaver + ErrorSmith preprocessing integration
- **Key functionality**:
  - Error profiling (ErrorProfiler)
  - K-mer prediction (AdaptiveKmerPredictor/K-Weaver)
  - Read correction (ErrorSmith technology-specific correctors)
  - Preprocessing result tracking and statistics

### 2. **AssemblyOrchestrator** (667 lines)
- **Former location**: `utils/pipeline_orchestrator.py`
- **Purpose**: Technology-specific assembly routing
- **Key functionality**:
  - Read-type detection (Illumina, HiFi, ONT, Ancient DNA, Mixed)
  - Deterministic assembly flows:
    - Illumina: OLC → DBG → String Graph (if UL) → Hi-C
    - HiFi/ONT: DBG → String Graph (if UL) → Hi-C
    - Ancient: DBG → String Graph (if UL) → Hi-C
  - UL anchor generation and graph conversion
  - Contig extraction and Hi-C scaffolding

### 3. **PipelineOrchestrator** (465 lines → 1,219 lines consolidated)
- **Location**: `utils/pipeline.py` (master file)
- **Purpose**: End-to-end pipeline execution with checkpointing
- **Key functionality**:
  - Step-by-step execution (Profile → Correct → Assemble → Finish)
  - Checkpoint creation and recovery
  - AI model loading and management
  - File I/O coordination

---

## Consolidated Architecture

The new **`PipelineOrchestrator`** (1,219 lines) combines all three coordinators:

```
PipelineOrchestrator (Master Orchestrator)
│
├── Preprocessing Methods (from PreprocessingCoordinator)
│   ├── Error profiling
│   ├── K-mer prediction (K-Weaver)
│   └── Read correction (ErrorSmith)
│
├── Assembly Methods (from AssemblyOrchestrator)
│   ├── Technology-specific routing
│   ├── OLC assembly (Illumina)
│   ├── DBG construction
│   ├── String graph overlay
│   ├── UL anchor generation
│   └── Hi-C scaffolding
│
└── Pipeline Control (original PipelineOrchestrator)
    ├── Step execution management
    ├── Checkpoint support
    ├── AI model loading
    └── File I/O operations
```

---

## Data Structures Integrated

The following data structures from `PreprocessingCoordinator` are now in `pipeline.py`:

- **`KmerPrediction`**: Predicted k-mer sizes for different assembly stages
- **`PreprocessingStats`**: Statistics from preprocessing run
- **`PreprocessingResult`**: Complete preprocessing pipeline result
- **`AssemblyResult`**: Result of assembly pipeline (DBG, string graph, contigs, scaffolds)

---

## Import Updates

### Updated Files:
1. **`strandweaver/utils/__init__.py`**
   - Removed: `from .pipeline_orchestrator import AssemblyOrchestrator`
   - Added exports: `AssemblyResult`, `KmerPrediction`, `PreprocessingStats`, `PreprocessingResult`

2. **`strandweaver/assembly_utils/__init__.py`**
   - Removed: `from .utilities import (PreprocessingCoordinator, AssemblyUtils)`
   - Updated docstring to note consolidation

3. **`strandweaver/cli.py`**
   - No changes needed (already imported from `utils.pipeline`)

---

## Archived Files

All three original coordinator files have been archived to `archive/coordinators_legacy/`:

1. **`pipeline_orchestrator.py`** (25 KB)
   - Original AssemblyOrchestrator

2. **`preprocessing_coordinator_utilities.py`** (16 KB)
   - Original PreprocessingCoordinator from `assembly_utils/utilities.py`

3. **`preprocessing_coordinator_utilities_duplicate.py`** (16 KB)
   - Duplicate PreprocessingCoordinator from `assembly/utilities.py`

---

## Deleted Files

The following files have been removed from the codebase (archived first):
- `strandweaver/utils/pipeline_orchestrator.py`
- `strandweaver/assembly_utils/utilities.py`
- `strandweaver/assembly/utilities.py`

---

## Key Principles Preserved

1. **GNN-First Architecture**: PathWeaver maintains GNN-first path prediction with algorithm fallback
2. **Deterministic Assembly Flows**: String graph ALWAYS follows DBG when UL reads are available
3. **Technology-Specific Processing**: Separate flows for Illumina, HiFi, ONT, Ancient DNA, and Mixed reads
4. **Checkpoint Support**: Full checkpoint/recovery capability for long-running pipelines
5. **AI Integration**: Lazy loading of AI models for adaptive k-mer selection and error correction

---

## Benefits of Consolidation

1. **Single Entry Point**: One coordinator for the entire pipeline instead of three
2. **Reduced Complexity**: Easier for users to understand and implement
3. **Eliminated Duplication**: Removed duplicate `utilities.py` files
4. **Better Maintainability**: All coordination logic in one place
5. **Cleaner Imports**: Simplified import structure across the codebase

---

## Pipeline Flow Summary

The consolidated `PipelineOrchestrator` executes the following flow:

```
1. Profile → Error profiling + k-mer prediction (K-Weaver)
2. Correct → Technology-specific read correction (ErrorSmith)
3. Assemble → Technology-specific assembly routing:
   - Illumina: OLC → DBG → String Graph → Hi-C
   - HiFi: DBG → String Graph → Hi-C
   - ONT: DBG → String Graph (+ UL overlay) → Hi-C
   - Ancient: DBG → String Graph → Hi-C
4. Finish → Polishing, gap filling, final output
```

---

## Verification

- ✅ All imports updated successfully
- ✅ No errors in consolidated `pipeline.py`
- ✅ No errors in `utils/__init__.py`
- ✅ No errors in `cli.py`
- ✅ Old files archived to `archive/coordinators_legacy/`
- ✅ Redundant files deleted
- ✅ Data structures properly integrated
- ✅ Assembly methods fully functional

---

## Usage Example

```python
from strandweaver.utils import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator(config, checkpoint_dir="checkpoints")

# Run complete pipeline
result = orchestrator.run(start_from=None, resume=False)

# Or run individual steps
orchestrator._step_profile()
orchestrator._step_correct()
orchestrator._step_assemble()
orchestrator._step_finish()
```

---

## Related Consolidations

This consolidation follows previous major consolidations:
1. **GNN Consolidation**: `gnn_path_predictor.py` + `gnn_models.py` → `pathweaver_module.py`
2. **Coordinator Consolidation**: `PreprocessingCoordinator` + `AssemblyOrchestrator` + `PipelineOrchestrator` → Unified `PipelineOrchestrator`

---

## Conclusion

The coordinator consolidation successfully merged 1,535 lines from three separate coordinators into a single, unified 1,219-line `PipelineOrchestrator` that manages the complete StrandWeaver assembly pipeline. This consolidation eliminates redundancy, simplifies the codebase, and provides a cleaner user experience with a single entry point for pipeline execution.
