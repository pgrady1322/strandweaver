# StrandWeaver Pipeline Flow

**Visual representation of the complete assembly pipeline**

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT DATA (Multi-Technology)                 │
├─────────────────────────────────────────────────────────────────┤
│  Illumina PE  │  HiFi  │  ONT  │  Ultra-Long  │  Hi-C  │  aDNA  │
└────────┬──────┴───┬────┴───┬────┴──────┬───────┴───┬────┴───┬───┘
         │          │        │           │           │        │
         ▼          ▼        ▼           ▼           ▼        ▼
┌────────────────────────────────────────────────────────────────┐
│                   ERROR CORRECTION (Technology-Aware)          │
├────────────────────────────────────────────────────────────────┤
│  • Illumina: k-mer based correction                            │
│  • ONT: Homopolymer-aware correction                           │
│  • Ancient DNA: Deamination modeling (C→T, G→A)                │
│  • PacBio: CCS-specific error handling                         │
└────────┬───────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────┐
│              ASSEMBLY STAGE 1: Contig Building                 │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Illumina Only:                                                 │
│  ┌──────────────────────────────────────────────┐              │
│  │  Overlap-Layout-Consensus (OLC)              │              │
│  │  • GPU-accelerated overlap detection         │              │
│  │  • Parallel processing (6-7× speedup)        │              │
│  │  • Creates initial contigs                   │              │
│  └─────────────────────┬────────────────────────┘              │
│                        │                                        │
│  Long Reads (HiFi/ONT):                                         │
│  ┌──────────────────────────────────────────────┐              │
│  │  De Bruijn Graph (DBG)                       │              │
│  │  • GPU k-mer extraction (∞× speedup)         │              │
│  │  • Graph construction (9-22s vs hanging)     │              │
│  │  • Unitig extraction                         │              │
│  └─────────────────────┬────────────────────────┘              │
│                        │                                        │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│         ASSEMBLY STAGE 2: Ultra-Long Read Integration          │
├────────────────────────────────────────────────────────────────┤
│  String Graph Construction (if UL reads available)              │
│  ┌──────────────────────────────────────────────┐              │
│  │  • Anchor-guided alignment (GraphAligner)    │              │
│  │  • GPU-accelerated UL mapping (15× speedup)  │              │
│  │  • Long-range connectivity                   │              │
│  │  • AI-powered UL routing (ambiguity          │              │
│  │    resolution)                                │              │
│  └─────────────────────┬────────────────────────┘              │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              ASSEMBLY STAGE 3: AI-Powered Refinement           │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌───────────────────┐  ┌─────────────┐ │
│  │ Edge AI Filter   │  │ GNN Path Predict  │  │ Diploid AI  │ │
│  │ (overlap class.) │  │ (optimal paths)   │  │ (haplotype) │ │
│  └─────────┬────────┘  └────────┬──────────┘  └──────┬──────┘ │
│            │                    │                     │        │
│            └────────────────────┼─────────────────────┘        │
│                                 ▼                              │
│                        ┌─────────────────┐                     │
│                        │  Graph Cleanup  │                     │
│                        │  • Remove tips  │                     │
│                        │  • Pop bubbles  │                     │
│                        │  • SV detection │                     │
│                        └────────┬────────┘                     │
└─────────────────────────────────┼──────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────┐
│           ASSEMBLY STAGE 4: Hi-C Scaffolding (Optional)        │
├────────────────────────────────────────────────────────────────┤
│  GPU-Accelerated Hi-C Integration (20-50× speedup)             │
│  ┌──────────────────────────────────────────────┐              │
│  │  1. Contact Matrix Building (20-40× faster)  │              │
│  │  2. Spectral Phasing (15-35× faster)         │              │
│  │  3. Edge Support Computation (8-12× faster)  │              │
│  │  4. Chromosome-scale scaffolds               │              │
│  └─────────────────────┬────────────────────────┘              │
└────────────────────────┼────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│              OPTIONAL: Manual Curation (BandageNG)             │
├────────────────────────────────────────────────────────────────┤
│  • Export graph in GFA format                                   │
│  • Visual inspection and editing                               │
│  • Import corrections back into pipeline                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│                      FINAL OUTPUT                               │
├────────────────────────────────────────────────────────────────┤
│  • Assembled contigs/scaffolds (FASTA)                          │
│  • Assembly graph (GFA)                                         │
│  • Quality metrics                                              │
│  • Error reports and visualizations                             │
└─────────────────────────────────────────────────────────────────┘
```

## Stage Details

### Stage 1: Error Correction
- **Input**: Raw sequencing reads
- **Technologies**: Illumina, HiFi, ONT, Ancient DNA
- **Methods**:
  - K-mer based correction
  - Homopolymer-aware (ONT)
  - Deamination modeling (aDNA)
- **Output**: Corrected reads

### Stage 2: Contig Building
- **OLC (Illumina)**:
  - Overlap detection (GPU-accelerated)
  - Layout construction
  - Consensus generation
- **DBG (HiFi/ONT)**:
  - K-mer extraction (GPU)
  - Graph construction (GPU)
  - Unitig extraction

### Stage 3: UL Integration
- **String Graph**:
  - Anchor finding (GPU, 15× faster)
  - GraphAligner mapping
  - AI-powered routing

### Stage 4: AI Refinement
- **Edge Classification**: Filter spurious overlaps
- **GNN Path Prediction**: Optimal contig paths
- **Diploid Disentangling**: Separate haplotypes
- **SV Detection**: Identify structural variants
- **Graph Cleanup**: Remove artifacts

### Stage 5: Hi-C Scaffolding
- **Contact Matrix**: GPU-accelerated (20-40× faster)
- **Spectral Phasing**: GPU clustering (15-35× faster)
- **Edge Support**: Vectorized computation (8-12× faster)
- **Scaffolding**: Chromosome-scale assembly

## GPU Acceleration Points

| Stage | Operation | Speedup | Backend Support |
|-------|-----------|---------|-----------------|
| Contig Building | Overlap detection | 7.2× | CUDA/MPS/CPU |
| DBG Construction | K-mer extraction | ∞× | CUDA/MPS/CPU |
| DBG Construction | Graph building | ∞× | CUDA/MPS/CPU |
| UL Integration | Anchor finding | 15× | CUDA/MPS/CPU |
| Hi-C | Contact matrix | 20-40× | CUDA/MPS/CPU |
| Hi-C | Spectral phasing | 15-35× | CUDA/MPS/CPU |
| Hi-C | Edge support | 8-12× | CUDA/MPS/CPU |

## Decision Points

### Technology Routing

```
Input reads → Technology detection → Route to appropriate method
│
├─ Illumina only → OLC → DBG (from contigs) → String Graph (if UL)
├─ HiFi/ONT → DBG directly → String Graph (if UL)
├─ Ancient DNA → Damage correction → OLC/DBG
└─ Ultra-long → String Graph extension
```

### AI Module Activation

```
Assembly graph → AI edge filter → Graph cleanup → AI path prediction → 
  Contigs → AI diploid disentangler → Haplotypes → Hi-C → Scaffolds
```

## Checkpointing System

Pipeline can be resumed from any stage:
1. Error correction complete
2. Contig building complete
3. UL integration complete
4. AI refinement complete
5. Hi-C scaffolding complete

## Performance Profile

**Typical Runtime (1 Mb genome, Illumina + HiFi + UL + Hi-C):**

| Stage | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| Error correction | ~5 min | ~5 min | 1× (not GPU-accelerated) |
| Contig building | ~6 min | ~50 sec | 7.2× |
| DBG construction | Never completes | ~15 sec | ∞ |
| UL integration | ~5 min | ~20 sec | 15× |
| AI refinement | ~3 min | ~3 min | 1× (CPU-bound currently) |
| Hi-C scaffolding | ~6 min | ~20 sec | 20-50× |
| **Total** | **~25+ min** | **~2-3 min** | **12-20×** |

## See Also

- [GPU Acceleration Guide](../GPU_ACCELERATION_GUIDE.md)
- [AI/ML Guide](AI_ML_GUIDE.md)
- [Assembly Module Reference](../ASSEMBLY_MODULE_REFERENCE.md)
- [Development Roadmap](MASTER_DEVELOPMENT_ROADMAP.md)
