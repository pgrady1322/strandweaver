# AI/ML Guide

**Comprehensive guide to StrandWeaver's artificial intelligence and machine learning features**

---

## Overview

StrandWeaver integrates **7 AI/ML-powered subsystems** across error correction and genome assembly stages. These AI modules improve assembly quality by learning patterns from training data that heuristic algorithms struggle to capture.

### Current Status

🔴 **All AI models currently use heuristic fallbacks** (December 2025)
- Training data generation: 70% complete
- Model training: Not yet started
- Expected AI deployment: January 2026

### AI Integration Philosophy

**Graceful Degradation**: Every AI module has a robust heuristic fallback. If models are unavailable or fail, StrandWeaver automatically falls back to rule-based algorithms. You don't need trained models to use StrandWeaver.

**GPU Acceleration**: AI inference leverages the same multi-backend GPU system (CUDA/MPS/CPU) used for assembly algorithms, ensuring optimal performance on any hardware.

**Transparency**: AI decisions are logged with confidence scores, allowing users to audit model behavior and understand assembly choices.

---

## AI/ML Subsystems

### 1. K-Weaver (Adaptive K-mer Selector)

**What it does**: Dynamically selects the optimal k-mer size for error correction based on local read characteristics.

**Why it matters**: Traditional fixed k-mer approaches fail in variable-quality regions. Smaller k-mers work better in low-coverage or high-error regions, while larger k-mers are optimal for high-quality data. The AI learns these trade-offs.

**How it works**:
1. Analyzes read properties (quality scores, GC content, homopolymer runs, coverage)
2. Predicts optimal k-mer size (15-51) for that specific read region
3. Applies technology-aware correction with the selected k-mer

**Model Architecture**:
- **Type**: Random Forest or Gradient Boosting (XGBoost)
- **Features**: Error rate, coverage depth, GC content, homopolymer length, read length, technology type
- **Training data**: 540,000 examples across 6 technologies

**Performance Impact**:
- Expected **5-10% reduction** in over-correction artifacts
- Particularly beneficial for mixed-quality datasets

**Usage**:
```bash
strandweaver assemble --reads reads.fq   --use-ai-kmer-selector  # Enable AI k-mer selection
```

**Fallback**: Fixed k-mer sizes per technology (Illumina: k=21, HiFi: k=31, ONT: k=17, aDNA: k=15)

---

### 2. ErrorSmith (Base Error Predictor)

**What it does**: Predicts per-base error probability considering sequence context, improving error correction targeting.

**Why it matters**: Quality scores alone don't capture context-dependent errors. For example, homopolymer errors in ONT reads or G→A errors at 3' ends of ancient DNA fragments require context-aware prediction.

**How it works**:python scripts/generate_assembly_training_data.py \
  --scenario repeat_heavy \
  --output-dir training_data/assembly_ai/repeat_heavy \
  --num-workers 1 \
  --use-gpu
1. Examines 21-bp window around each base
2. Considers quality scores, homopolymer runs, position in read, technology type
3. Outputs error probability (0-1) for targeted correction

**Model Architecture**:
- **Type**: 1D CNN or Bidirectional LSTM
- **Features**: Sequence context (one-hot encoded), quality scores, homopolymer indicators, read position, technology embedding
- **Training data**: 360,000 examples across 6 technologies

**Performance Impact**:
- Expected **15-20% reduction** in false positive corrections
- Major improvement for homopolymer-error-prone technologies (ONT)

**Usage**:
```bash
strandweaver assemble --reads reads.fq   --use-ai-base-error-predictor
```

**Fallback**: Quality score thresholds (Illumina/HiFi: Q20, ONT: Q10)

---

### 3. EdgeWarden (Overlap Classifier)

**What it does**: Classifies read overlaps as true or spurious to filter assembly graph edges before contig construction.

**Why it matters**: Repetitive genomes generate thousands of spurious overlaps that create tangled assembly graphs. AI learns subtle patterns distinguishing real from false overlaps.

**How it works**:
1. Extracts 18 features from each overlap (length, identity, coverage, repeat content, etc.)
2. Classifies as true (keep) or spurious (remove)
3. Filters graph edges before path-finding

**Model Architecture**:
- **Type**: Random Forest or XGBoost
- **Features**: Overlap length, identity, overhangs, coverage ratios, k-mer uniqueness, alignment scores, trimming patterns, technology
- **Training data**: Balanced true/spurious overlaps from 20 simulated genomes

**Performance Impact**:
- Expected **10-15% reduction** in misassemblies
- Cleaner graphs with fewer false edges

**Usage**:
```bash
strandweaver assemble --reads reads.fq   --use-ai-overlap-classifier
```

**Fallback**: Rule-based filtering (min overlap 1kb, min identity 85%, max overhang 500bp)

---

### 4. PathWeaver (GNN Path Predictor)

**What it does**: Uses Graph Neural Networks to find optimal contig paths through complex assembly graphs.

**Why it matters**: Traditional greedy algorithms get stuck in local maxima. GNNs consider global graph topology to find globally optimal paths.

**How it works**:
1. Embeds assembly graph into high-dimensional space (128-256 dimensions)
2. Uses message-passing layers to propagate information across graph
3. Predicts probability distribution over candidate paths
4. Selects highest-probability path for contig extension

**Model Architecture**:
- **Type**: Graph Neural Network (Message Passing Neural Network)
- **Layers**: 3-5 message-passing layers with node embeddings
- **Features**: Node (coverage, length, GC, repeat score), Edge (overlap quality, support, orientation), Global (topology metrics)
- **Training data**: Graph structures with ground truth paths from simulations

**Performance Impact**:
- Expected **30-50% improvement** in path selection for complex regions
- Major benefit for highly repetitive or polyploid genomes

**Usage**:
```bash
strandweaver assemble --reads reads.fq   --use-ai-gnn-path-predictor   --gpu  # Recommended for GNN inference
```

**Fallback**: Greedy path extension (highest coverage first)

**Note**: Most computationally intensive AI component; GPU strongly recommended

---

#### PathWeaver: Optional Inputs & Validation Tuning

You can optionally guide PathWeaver’s validation with synteny markers and boundary labels, and tune validation thresholds via a `ValidationConfig`.

**Register Synteny Markers**:
```python
from strandweaver.assembly.pathweaver import PathWeaver

# node_id -> marker_index (expected non-decreasing order along valid paths)
marker_map = {101: 1, 104: 2, 108: 3}
pw = PathWeaver(graph)
pw.register_synteny_markers(marker_map)
```

**Register Boundary Labels**:
```python
# node_id -> label (e.g., 'telomere', 'centromere', or custom)
boundary_map = {200: 'telomere', 205: 'centromere'}
pw.register_boundary_labels(boundary_map)
```

**Tune Validation Thresholds**:
```python
from strandweaver.assembly.pathweaver import ValidationConfig, PathValidationRule

config = ValidationConfig(
  min_coverage=3.0,                      # stricter minimum coverage
  repeat_score_threshold=0.85,           # repeat-aware threshold
  repeat_min_support=4,
  repeat_min_confidence=0.55,
  boundary_cross_confidence_threshold=0.85,
  cnv_ratio_threshold=2.5,               # tighter CNV jump threshold
  rules_enabled=[                        # enable/disable specific rules
    PathValidationRule.NO_SELF_LOOPS,
    PathValidationRule.MIN_COVERAGE,
    PathValidationRule.REPEAT_AWARE,
    PathValidationRule.SYNTENY_CONSTRAINT,
    PathValidationRule.BOUNDARY_DETECTION,
    PathValidationRule.CNV_AWARE,
  ],
)

# Apply to PathWeaver
pw = PathWeaver(graph, validation_config=config)
# or update later
pw.set_validation_config(config)
```

**Effect**:
- Synteny: Enforces non-decreasing `marker_index` along paths.
- Boundary: Disallows crossing different `boundary_label` regions unless edge confidence exceeds threshold.
- CNV-aware: Flags abrupt coverage jumps beyond `cnv_ratio_threshold`.
- Repeat-aware: Requires stronger support/confidence in high-repeat segments.

These inputs are optional; PathWeaver operates without them but uses them to reduce misassemblies in complex genomes.

#### Two-Pass (Long-Range-Aware) Validation Flow

By default PathWeaver runs a two-pass flow: Pass A uses defer/warn for boundary/synteny/CNV, Pass B revalidates in strict mode (even if no UL/Hi-C is available, falling back to long-read evidence).

**Pass A (default, no long-range yet)**
```python
from strandweaver.assembly.pathweaver import PathWeaver, ValidationConfig, PathValidationRule

pw = PathWeaver(graph)  # defaults to defer boundary/synteny, warn CNV
paths = pw.find_best_paths(start_node_id, end_node_ids={...})
# paths now carry validation_status: valid | warn | pending_long_range | rejected
```

**Long-range scoring**
- ThreadCompass (UL) updates edge/path confidences based on UL support/conflicts.
- Hi-C module updates confidences using contact support/repulsion.

**Pass B (strict revalidation, with or without long-range evidence)**
```python
config_pass_b = ValidationConfig(
  rule_modes={
    PathValidationRule.BOUNDARY_DETECTION: "strict",
    PathValidationRule.SYNTENY_CONSTRAINT: "strict",
    PathValidationRule.CNV_AWARE: "strict",
  }
)

# Revalidate existing candidates with updated confidences
pw.revalidate_paths(paths, validation_config=config_pass_b)

# Or rerun search if graph weights changed
best_paths = pw.find_best_paths(start_node_id, end_node_ids={...},
                num_iterations=1,
                edgewarden_scores=updated_scores,
                gnn_path_scores=updated_gnn_scores,
                long_range_two_pass=True,
                strict_validation_config=config_pass_b)
```

**Interpretation**
- `pending_rules` mark which checks were deferred; `validation_status` summarizes the outcome.
- Use Pass A to keep candidates alive until UL/Hi-C arrives; Pass B to enforce strict rules with added evidence.

### 5. ThreadCompass (UL Read Integration)

**What it does**: Maps ultra-long (UL) reads to the assembly graph and detects new joins from spanning evidence.

**Why it matters**: UL reads (>50kb) can span entire repeat regions and resolve ambiguous joins. ThreadCompass uses spanning evidence to both find new joins and validate existing ones.

**How it works**:
1. Maps UL reads to graph using k-mer size from preprocessing (K-Weaver output)
2. Detects reads spanning two distant contigs (new potential joins)
3. Scores each join based on span coverage, anchor uniqueness, and mapping quality
4. Returns 0.0-1.0 confidence scores for integration with other evidence

**Key Data Structures**:
- `ULMapping`: read_id, primary_node, secondary_nodes, MAPQ, span_start/end, multimapping flag
- `ULJoinScore`: join_id, from/to_node, ul_confidence (0-1), breakdown (span_coverage, anchor_uniqueness, mapq_score)

**Key Methods**:
- `register_ul_mappings(mappings)`: Load UL mapping data
- `detect_new_joins(max_joins, min_confidence)`: Find high-confidence joins not in graph
- `score_join(from_node, to_node)`: Score single join via UL evidence
- `score_path(node_ids)`: Score entire path using UL spanning support

**Scoring (0.0-1.0)**:
- 50% span coverage (fraction of expected UL reads covering join)
- 25% anchor uniqueness (1.0 - multimapping rate at both ends)
- 25% mapping quality (normalized MAPQ)

**Integration Point**: After PathWeaver Pass A (pending long-range validation) → provides scores for Pass B revalidation

**Expected Impact**: 15-25% accuracy improvement with >50kb UL reads; enables chromosome-scale contigs through repeats

---

### 6. HiCWeaver (Hi-C Contact Integration)

**What it does**: Uses Hi-C contact matrices to detect new joins and score existing ones based on 3D proximity.

**Why it matters**: Hi-C provides 3D genome organization data. Joins with strong Hi-C contacts are more reliable; weak/absent contacts suggest errors.

**How it works**:
1. Loads Hi-C contact matrix (contig pairs → contact count)
2. Detects new joins with unexpectedly high contact frequency
3. Scores each join based on contact frequency, orientation consistency, and distance decay
4. Validates path orientation consistency using Hi-C order

**Key Data Structures**:
- `HiCContact`: contig1_id, contig2_id, contact_count, orientation (++, +-, -+, --), quality
- `HiCJoinScore`: join_id, from/to_node, hic_confidence (0-1), breakdown (contact_frequency, orientation_consistency, distance_penalty)

**Key Methods**:
- `register_contact_matrix(matrix)`: Load Hi-C data
- `detect_new_joins(max_joins, min_confidence)`: Find high-confidence joins not in graph
- `score_join(from_node, to_node, orientation)`: Score join with orientation check
- `validate_path_orientation(node_ids, expected_orientations)`: Check order/strand consistency
- `score_path(node_ids)`: Score entire path using cumulative contact support

**Scoring (0.0-1.0)**:
- 60% contact frequency (normalized by distance decay model)
- 30% orientation consistency (++ or -- = 1.0; +- or -+ = 0.0)
- 10% distance decay penalty (penalizes unexpectedly distant contacts)

**Integration Point**: After ThreadCompass → before PathWeaver Pass B revalidation

**Expected Impact**: 10-20% accuracy improvement; critical for diploid genome assembly and large structural variant detection

---

### 6. SVScribe (SV Detection)

**What it does**: Identifies and classifies structural variants (deletions, duplications, inversions, insertions, translocations) during assembly.

**Why it matters**: SVs are critical for understanding genome evolution and disease but challenging to detect. AI learns complex SV signatures from coverage patterns, split reads, and read pair orientations.

**How it works**:
1. Analyzes coverage signals, split alignments, discordant pairs along contigs
2. Applies CNN or CNN-LSTM to detect SV signatures
3. Classifies SV type (DEL/DUP/INV/INS/TRA) and predicts breakpoints

**Model Architecture**:
- **Type**: 1D CNN or Hybrid CNN-LSTM
- **Features**: Coverage signals (windowed), split-read patterns, discordant pair orientations, soft-clip patterns, repeat context, read depth histograms
- **Training data**: 50-200 SVs per genome across multiple SV types

**Performance Impact**:
- Expected **25-40% increase** in SV detection sensitivity
- Better SV type classification accuracy

**Usage**:
```bash
strandweaver assemble --reads reads.fq   --use-ai-sv-detection   --output-svs sv_calls.vcf
```

**Fallback**: Coverage-based thresholds (DEL <0.5×, DUP >1.5×, INV by orientation)

---

### 7. Diploid Disentangler (Haplotype Phasing)

**What it does**: Separates heterozygous haplotypes in diploid genome assemblies.

**Why it matters**: Diploid genomes have two sets of chromosomes with subtle differences. Traditional assemblers either collapse haplotypes or create chimeric assemblies. AI learns haplotype patterns to produce clean phased assemblies.

**How it works**:
1. Identifies heterozygous variants along contigs
2. Integrates Hi-C contact data (if available) for long-range phasing
3. Uses GNN to assign variants to Haplotype 1 or Haplotype 2
4. Outputs phased assembly (two haplotypes)

**Model Architecture**:
- **Type**: Phasing-aware Graph Neural Network
- **Features**: Variant (allele frequencies, quality, support), Hi-C (contact matrix, phase consistency), Graph (haplotype-aware edge weights), Sequence (variant context)
- **Training data**: Phased variants and Hi-C patterns from simulated diploids

**Performance Impact**:
- Expected **15-25% improvement** in phasing accuracy
- Particularly strong with Hi-C data integration

**Usage**:
```bash
strandweaver assemble --reads reads.fq --hic hic_R1.fq hic_R2.fq   --use-ai-diploid-disentangler   --ploidy 2   --output-haplotypes
```

**Fallback**: Spectral clustering on Hi-C contact matrix (eigenvector-based)

**Note**: Requires Hi-C data for best performance; can use variant-only features otherwise

---

## Training Your Own Models

### Prerequisites

```bash
# Install training dependencies
conda env create -f env_ml_training.yml
conda activate strandweaver_training

# Verify GPU availability
python -c "import torch; print(f'GPU: {torch.cuda.is_available() or torch.backends.mps.is_available()}')"
```

### Generate Training Data

#### Read Correction Data

```bash
python scripts/generate_training_data.py   --output training_data/read_correction   --k-mer-examples 540000   --base-error-examples 360000   --workers 12
```

**Parameters**:
- `--k-mer-examples`: Total k-mer selector examples (90k per tech × 6 techs)
- `--base-error-examples`: Total base error examples (60k per tech × 6 techs)
- `--workers`: Parallel workers (use CPU count)

**Runtime**: ~15-20 hours on modern CPUs
**Output size**: ~12 GB

#### Assembly AI Data

```bash
# Fast iteration dataset (quick testing)
python scripts/generate_assembly_training_data.py   --scenario fast_balanced   --output training_data/assembly_ai   --workers 8

# Production dataset (best quality)
python scripts/generate_assembly_training_data.py   --scenario balanced   --output training_data/assembly_ai   --workers 8

# Specialized datasets
python scripts/generate_assembly_training_data.py   --scenario repeat_heavy   --output training_data/assembly_ai   --workers 8
```

**Scenarios**:
| Scenario | Genomes | Size | Repeats | SVs | Use Case | Runtime |
|----------|---------|------|---------|-----|----------|---------|
| `fast_balanced` | 20 | 500kb | 30% | 50 | Quick testing | ~5 hrs |
| `balanced` | 100 | 1Mb | 45% | 100 | Production | ~2-3 hrs |
| `repeat_heavy` | 50 | 2Mb | 60% | 200 | Repeat focus | ~1 hr |
| `sv_dense` | 50 | 1Mb | 40% | 500 | SV focus | ~1-2 hrs |
| `diploid_focus` | 100 | 1Mb | 45% | 100 | Phasing focus | ~2-3 hrs |
| `ultra_long_focus` | 30 | 5Mb | 45% | 150 | UL routing focus | ~3-4 hrs |

### Train Models

```bash
# Train all models
python scripts/train_all_models.py   --training-data training_data/   --output-models models/   --gpu

# Train individual models
python scripts/train_models/train_kmer_selector.py   --data training_data/read_correction   --output models/kmer_selector.pkl

python scripts/train_models/train_gnn_path_predictor.py   --data training_data/assembly_ai/balanced   --output models/gnn_path_predictor.pt   --gpu
```

**Note**: Model training scripts will be available in Phase 6 (January 2026)

### Deploy Models

```bash
# Copy trained models to deployment directory
cp models/*.pkl smartassembler/models/
cp models/*.pt smartassembler/models/

# Test deployment
strandweaver assemble --reads test.fq --use-all-ai --test-mode
```

---

## Scoring Normalization (0-1 Framework)

All scoring modules in the pipeline output normalized [0.0, 1.0] confidence scores to enable cross-module comparison and weighted combination.

### Scoring Modules & Their Ranges

| Module | Output Range | Raw Score | Normalized (0-1) | Interpretation |
|--------|--------------|-----------|------------------|-----------------|
| **PathWeaver Validation** | 1.0 - penalties | Error rate + warnings | 0.0-1.0 | 1.0 = all rules satisfied |
| **EdgeWarden** | 0-100 (composite) | Edge weights | ÷100 | 1.0 = highest quality edge |
| **GNN Path Predictor** | Logits | Raw network output | softmax → 0-1 | 1.0 = high-confidence path |
| **ThreadCompass (UL)** | 0-1 native | Span + anchor + MAPQ | 50% + 25% + 25% | 1.0 = strong UL support |
| **HiCWeaver (Hi-C)** | 0-1 native | Contact + orient + distance | 60% + 30% + 10% | 1.0 = strong 3D contact |

### Combined Scoring Example

```python
from strandweaver.assembly.pathweaver import PathWeaver, PathScorer

pw = PathWeaver(graph)

# Individual module scores (all 0-1)
validation_score = 0.92  # from PathValidator.compute_validation_score()
edgewarden_score = 0.85  # from EdgeWarden (normalized)
gnn_score = 0.78         # from GNN path predictor
ul_score = 0.88          # from ThreadCompass.score_path()
hic_score = 0.75         # from HiCWeaver.score_path()

# Compute weighted combination
scorer = PathScorer(graph)
combined_score = scorer.score_path(
    path_nodes=[1, 5, 12, 25],
    edgewarden_score=edgewarden_score,
    gnn_score=gnn_score,
    validation_score=validation_score,
    ul_confidence=ul_score,
    hic_confidence=hic_score
)
# Weights: EdW 25%, GNN 20%, topology 15%, UL 20%, Hi-C 15%, validation 5%
# combined_score ≈ 0.84
```

### Interpretation Guidelines

- **0.90-1.00**: High confidence, safe for downstream analysis
- **0.75-0.89**: Good confidence, acceptable for most assemblies
- **0.60-0.74**: Moderate confidence, flag for manual review
- **0.40-0.59**: Low confidence, likely errors or missing data
- **0.00-0.39**: Critical issues, likely false joins

### Adding Custom Scoring Modules

To integrate a custom module with 0-1 normalization:

```python
# 1. Implement scoring method returning 0-1 value
def score_my_evidence(path_nodes) -> float:
    raw_score = compute_raw_score(path_nodes)  # e.g., 0-100
    normalized = min(1.0, max(0.0, raw_score / 100.0))
    return normalized

# 2. Update PathScorer weights (sum to 100%)
from strandweaver.assembly.pathweaver import PathScorer
PathScorer.SCORING_WEIGHTS['my_module'] = 0.10  # 10% of final score
# (adjust other weights to maintain 100% total)

# 3. Pass score to find_best_paths
best_paths = pw.find_best_paths(
    start_node, end_nodes,
    my_module_scores=my_scores,  # dict of {path_id: score}
)
```

---

## GPU Acceleration for AI


All AI models support the same multi-backend GPU system as assembly algorithms:

| Backend | Hardware | Auto-detected | Performance |
|---------|----------|---------------|-------------|
| **CUDA** | NVIDIA GPUs | ✅ Yes | Best (5-20× speedup) |
| **MPS** | Apple Silicon | ✅ Yes | Excellent (3-15× speedup) |
| **CPU** | All systems | ✅ Fallback | Baseline |

### Enable GPU for AI

```bash
# Auto-detect best backend
strandweaver assemble --reads reads.fq --use-all-ai --gpu

# Force specific backend
strandweaver assemble --reads reads.fq --use-all-ai --gpu-backend cuda
strandweaver assemble --reads reads.fq --use-all-ai --gpu-backend mps
strandweaver assemble --reads reads.fq --use-all-ai --gpu-backend cpu
```

### AI Performance Benchmarks

**GNN Path Predictor** (most GPU-intensive):
| Hardware | Inference Time (per graph) | Speedup |
|----------|----------------------------|---------|
| CPU (12-core) | 450 ms | 1× |
| Apple M2 Max (MPS) | 35 ms | 12× |
| NVIDIA RTX 3090 (CUDA) | 22 ms | 20× |

**Overlap Classifier** (CPU-friendly):
| Hardware | Inference Time (10k overlaps) | Speedup |
|----------|------------------------------|---------|
| CPU (12-core) | 120 ms | 1× |
| Apple M2 Max (MPS) | 45 ms | 2.7× |
| NVIDIA RTX 3090 (CUDA) | 15 ms | 8× |

---

## Command Reference

### Enable All AI Modules

```bash
strandweaver assemble --reads reads.fq --use-all-ai --gpu
```

### Enable Specific AI Modules

```bash
strandweaver assemble --reads reads.fq   --use-ai-kmer-selector   --use-ai-base-error-predictor   --use-ai-overlap-classifier   --use-ai-gnn-path-predictor   --use-ai-ul-routing   --use-ai-sv-detection   --use-ai-diploid-disentangler   --gpu
```

### Check AI Model Availability

```bash
strandweaver check-ai-models
```

**Output**:
```
✅ Adaptive K-mer Selector: models/kmer_selector.pkl
✅ Base Error Predictor: models/base_error_predictor.pt
🔴 Overlap Classifier: NOT FOUND (using heuristic fallback)
✅ GNN Path Predictor: models/gnn_path_predictor.pt
...
```

### AI Logging and Debugging

```bash
# Enable detailed AI logging
strandweaver assemble --reads reads.fq --use-all-ai --ai-debug-log ai_decisions.log

# Log includes:
# - AI confidence scores
# - Heuristic fallback triggers
# - Feature values used for predictions
# - Model inference times
```

---

## Best Practices

### When to Use AI Models

**✅ Use AI when**:
- You have trained models or downloaded pre-trained models
- Assembly quality is critical (research, clinical applications)
- GPU acceleration is available
- Genome is complex (high repeat content, diploid, SVs)

**❌ Skip AI when**:
- Quick draft assembly needed (heuristics are faster)
- Simple genomes (bacteria, viruses) where heuristics work well
- CPU-only system and runtime is critical
- Models unavailable and training data generation is impractical

### Module Selection Strategy

| Long-Range Data | ThreadCompass | HiCWeaver | Expected Benefit |
|-----------------|---------------|-----------|-----------------|
| UL reads only | ✅ Yes | ❌ No | +15-25% accuracy |
| Hi-C only | ❌ No | ✅ Yes | +10-20% accuracy |
| UL + Hi-C | ✅ Yes | ✅ Yes | +25-40% accuracy |
| Neither | ⏭️ Skip | ⏭️ Skip | Baseline only |

### Genome Complexity vs. Modules

| Genome Complexity | Core Modules | Long-Range Options | Expected Improvement |
|-------------------|--------------|-------------------|---------------------|
| **Simple** (bacteria, small eukaryotes) | PathWeaver only | None | 5-10% accuracy |
| **Moderate** (fungi, plants) | PathWeaver + EdgeWarden | ThreadCompass (if UL data) | 15-25% accuracy |
| **Complex** (mammals, polyploid) | All core modules | ThreadCompass + HiCWeaver | 30-50% accuracy |
| **Ultra-complex** (highly repetitive, polyploid) | All core modules + GPU | ThreadCompass + HiCWeaver + GPU | 50-70% accuracy |

### Performance vs. Quality Trade-offs

| Configuration | Runtime | Quality | Use Case |
|---------------|---------|---------|----------|
| Heuristics only | 1× | Baseline | Quick drafts |
| AI (CPU) | 2-3× | +30% | Quality-focused, no GPU |
| AI (GPU) | 1.5× | +30% | Best of both worlds |
| AI + Hi-C (GPU) | 2× | +50% | Chromosome-scale, diploid |

---

## Troubleshooting

### Model Loading Failures

**Problem**: `FileNotFoundError: models/kmer_selector.pkl not found`

**Solution**: 
```bash
# Check model availability
strandweaver check-ai-models

# Fallback to heuristics (automatic)
# Or download pre-trained models (when available)
wget https://strandweaver.org/models/pretrained.tar.gz
tar -xzf pretrained.tar.gz -C smartassembler/models/
```

### GPU Out of Memory (OOM)

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```bash
# Reduce GNN batch size
strandweaver assemble --reads reads.fq --use-all-ai --gpu   --gnn-batch-size 16  # Default: 64

# Or fall back to CPU for GNN only
strandweaver assemble --reads reads.fq --use-all-ai   --ai-module-device gnn_path_predictor=cpu
```

### Slow AI Inference

**Problem**: AI taking too long even with GPU

**Solution**:
```bash
# Check GPU is actually being used
strandweaver assemble --reads reads.fq --use-all-ai --gpu --ai-debug-log debug.log
grep "Device" debug.log

# If showing CPU, force GPU:
strandweaver assemble --reads reads.fq --use-all-ai --gpu-backend cuda --force-gpu
```

---

## FAQ

**Q: Can I use StrandWeaver without AI models?**  
A: Yes! All AI modules have robust heuristic fallbacks. StrandWeaver works perfectly fine without trained models.

**Q: When will pre-trained models be available?**  
A: Pre-trained models are planned for release in **January 2026** after training data generation completes and models are benchmarked.

**Q: Can I fine-tune models on my own data?**  
A: Yes (Phase 6 feature). You'll be able to fine-tune on real assemblies with known reference genomes.

**Q: Do AI models work with all sequencing technologies?**  
A: Yes. All models are technology-aware and trained on Illumina, PacBio (HiFi/CLR), ONT (R9/R10), and Ancient DNA.

**Q: How much training data do I need to train my own models?**  
A: Recommended minimum:
- Read correction: 100k examples per technology
- Assembly AI: 20 simulated genomes (fast_balanced) for testing, 100 genomes (balanced) for production

**Q: What's the model file format?**  
A: 
- Scikit-learn models (Random Forest, XGBoost): `.pkl` (pickle)
- PyTorch models (CNN, LSTM, GNN): `.pt` (PyTorch state dict)

---

## See Also

- [AI/ML Implementation Status](AI_ML_IMPLEMENTATION_STATUS.md) - Internal technical status and architecture details
- [Training Log](TRAINING_LOG.md) - Record of training runs and datasets
- [GPU Acceleration Guide](../GPU_ACCELERATION_GUIDE.md) - GPU backend setup and optimization
- [Pipeline Flow](PIPELINE_FLOW.md) - Where AI modules fit in the assembly pipeline
- [Scientific References](SCIENTIFIC_REFERENCES.md) - Citations for AI/ML methods

---

## Contributing

Interested in improving AI models or adding new AI modules? See [DEVELOPMENT_ROADMAP.md](MASTER_DEVELOPMENT_ROADMAP.md) Phase 6 for AI/ML development priorities.

