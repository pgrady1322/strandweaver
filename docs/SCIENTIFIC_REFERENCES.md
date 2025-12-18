# Scientific References

**Citations and sources for algorithms, methods, and data used in StrandWeaver**

---

## Assembly Algorithms

### Overlap-Layout-Consensus (OLC)
- Myers, E. W. (2005). "The fragment assembly string graph." *Bioinformatics*, 21(Suppl 2), ii79-ii85.
- Myers, E. W., et al. (2000). "A whole-genome assembly of Drosophila." *Science*, 287(5461), 2196-2204.

### De Bruijn Graph (DBG)
- Pevzner, P. A., Tang, H., & Waterman, M. S. (2001). "An Eulerian path approach to DNA fragment assembly." *Proceedings of the National Academy of Sciences*, 98(17), 9748-9753.
- Bankevich, A., et al. (2012). "SPAdes: A new genome assembly algorithm and its applications to single-cell sequencing." *Journal of Computational Biology*, 19(5), 455-477.

### String Graph
- Myers, E. W. (2005). "The fragment assembly string graph." *Bioinformatics*, 21(Suppl 2), ii79-ii85.
- Simpson, J. T., & Durbin, R. (2010). "Efficient construction of an assembly string graph using the FM-index." *Bioinformatics*, 26(12), i367-i373.

---

## Sequencing Technologies

### Oxford Nanopore (ONT)
- **ONT R9.4.1 Error Rates**:
  - Quick, J., et al. (2016). "Real-time, portable genome sequencing for Ebola surveillance." *Nature*, 530(7589), 228-232.
  - Average error rate: ~10-15%
  - Predominant errors: Insertions/deletions, homopolymer errors

- **ONT R10.4.1 Error Rates**:
  - Oxford Nanopore Technologies (2022). "R10.4.1 Pore Release Notes."
  - Average error rate: ~3-5%
  - Significant improvement in homopolymer accuracy

- **Homopolymer Error Characteristics**:
  - Delahaye, C., & Nicolas, J. (2021). "Sequencing DNA with nanopores: Troubles and biases." *PLoS ONE*, 16(10), e0257521.
  - Error rate increases with homopolymer length (exponential)

### PacBio
- **HiFi Reads (Circular Consensus Sequencing)**:
  - Wenger, A. M., et al. (2019). "Accurate circular consensus long-read sequencing improves variant detection and assembly of a human genome." *Nature Biotechnology*, 37(10), 1155-1162.
  - Average error rate: ~0.1-0.5% (Q20+)
  - Near-random error distribution

- **CLR (Continuous Long Reads)**:
  - Rhoads, A., & Au, K. F. (2015). "PacBio sequencing and its applications." *Genomics, Proteomics & Bioinformatics*, 13(5), 278-289.
  - Average error rate: ~10-15%
  - Predominantly insertion/deletion errors

### Illumina
- Glenn, T. C. (2011). "Field guide to next-generation DNA sequencers." *Molecular Ecology Resources*, 11(5), 759-769.
- Average error rate: ~0.1-1%
- Predominant errors: Substitutions (quality drops toward read ends)

### Ancient DNA (aDNA)
- **Deamination Patterns**:
  - Briggs, A. W., et al. (2007). "Patterns of damage in genomic DNA sequences from a Neandertal." *Proceedings of the National Academy of Sciences*, 104(37), 14616-14621.
  - C→T transitions at 5' ends
  - G→A transitions at 3' ends
  - Exponential increase near fragment ends

- **Fragment Length Distribution**:
  - Sawyer, S., et al. (2012). "Temporal patterns of nucleotide misincorporations and DNA fragmentation in ancient DNA." *PLoS ONE*, 7(3), e34131.
  - Mean fragment length: 35-80 bp (highly degraded samples)
  - Exponential decay distribution

---

## Hi-C Scaffolding

### Contact Matrix and Phasing
- Burton, J. N., et al. (2013). "Chromosome-scale scaffolding of de novo genome assemblies based on chromatin interactions." *Nature Biotechnology*, 31(12), 1119-1125.
- Lieberman-Aiden, E., et al. (2009). "Comprehensive mapping of long-range interactions reveals folding principles of the human genome." *Science*, 326(5950), 289-293.

### Spectral Clustering
- Ng, A. Y., Jordan, M. I., & Weiss, Y. (2002). "On spectral clustering: Analysis and an algorithm." *Advances in Neural Information Processing Systems*, 14, 849-856.
- Applied to Hi-C contact matrices for haplotype phasing

---

## Error Correction

### K-mer Based Correction
- Kelley, D. R., Schatz, M. C., & Salzberg, S. L. (2010). "Quake: quality-aware detection and correction of sequencing errors." *Genome Biology*, 11(11), R116.
- Salmela, L., & Schroder, J. (2011). "Correcting errors in short reads by multiple alignments." *Bioinformatics*, 27(11), 1455-1461.

### Technology-Specific Correction
- **ONT Homopolymer Correction**:
  - Shafin, K., et al. (2020). "Nanopore sequencing and the Shasta toolkit enable efficient de novo assembly of eleven human genomes." *Nature Biotechnology*, 38(9), 1044-1053.

- **Ancient DNA Damage Correction**:
  - Ginolhac, A., et al. (2011). "mapDamage: testing for damage patterns in ancient DNA sequences." *Bioinformatics*, 27(15), 2153-2155.
  - Jónsson, H., et al. (2013). "mapDamage2.0: fast approximate Bayesian estimates of ancient DNA damage parameters." *Bioinformatics*, 29(13), 1682-1684.

---

## Graph Neural Networks (GNN)

### GNN for Assembly Graphs
- Scarselli, F., et al. (2008). "The graph neural network model." *IEEE Transactions on Neural Networks*, 20(1), 61-80.
- Kipf, T. N., & Welling, M. (2017). "Semi-supervised classification with graph convolutional networks." *International Conference on Learning Representations (ICLR)*.

### GNN Applications in Bioinformatics
- Zhou, J., et al. (2020). "Graph neural networks: A review of methods and applications." *AI Open*, 1, 57-81.
- Applied to assembly graph path prediction and edge classification

---

## GPU Acceleration

### CUDA Programming
- Sanders, J., & Kandrot, E. (2010). *CUDA by Example: An Introduction to General-Purpose GPU Programming*. Addison-Wesley Professional.

### Apple Silicon MPS (Metal Performance Shaders)
- Apple Developer Documentation (2023). "Metal Performance Shaders." 
- https://developer.apple.com/documentation/metalperformanceshaders

### Bioinformatics GPU Applications
- Nobile, M. S., et al. (2017). "Graphics processing units in bioinformatics, computational biology and systems biology." *Briefings in Bioinformatics*, 18(5), 870-885.
- Liu, Y., Schmidt, B., & Maskell, D. L. (2012). "CUSHAW: a CUDA compatible short read aligner to large genomes based on the Burrows–Wheeler transform." *Bioinformatics*, 28(14), 1830-1837.

---

## Structural Variant (SV) Detection

### SV Calling Methods
- Sedlazeck, F. J., et al. (2018). "Accurate detection of complex structural variations using single-molecule sequencing." *Nature Methods*, 15(6), 461-468.
- Chaisson, M. J., et al. (2019). "Multi-platform discovery of haplotype-resolved structural variation in human genomes." *Nature Communications*, 10(1), 1784.

### Long-Read SV Detection
- Cretu Stancu, M., et al. (2017). "Mapping and phasing of structural variation in patient genomes using nanopore sequencing." *Nature Communications*, 8(1), 1326.

---

## Machine Learning for Genomics

### Random Forests and Gradient Boosting
- Breiman, L. (2001). "Random forests." *Machine Learning*, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

### CNNs for Sequence Analysis
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). "Deep learning." *Nature*, 521(7553), 436-444.
- Poplin, R., et al. (2018). "A universal SNP and small-indel variant caller using deep neural networks." *Nature Biotechnology*, 36(10), 983-987.

### LSTMs for Biological Sequences
- Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." *Neural Computation*, 9(8), 1735-1780.
- Quang, D., & Xie, X. (2016). "DanQ: a hybrid convolutional and recurrent deep neural network for quantifying the function of DNA sequences." *Nucleic Acids Research*, 44(11), e107.

---

## Assembly Quality Assessment

### BUSCO (Benchmarking Universal Single-Copy Orthologs)
- Simão, F. A., et al. (2015). "BUSCO: assessing genome assembly and annotation completeness with single-copy orthologs." *Bioinformatics*, 31(19), 3210-3212.
- Manni, M., et al. (2021). "BUSCO update: novel and streamlined workflows along with broader and deeper phylogenetic coverage for scoring of eukaryotic, prokaryotic, and viral genomes." *Molecular Biology and Evolution*, 38(10), 4647-4654.

### QUAST
- Gurevich, A., et al. (2013). "QUAST: quality assessment tool for genome assemblies." *Bioinformatics*, 29(8), 1072-1075.

---

## Software and Tools

### GraphAligner
- Rautiainen, M., & Marschall, T. (2020). "GraphAligner: rapid and versatile sequence-to-graph alignment." *Genome Biology*, 21(1), 253.

### BandageNG
- Wick, R. R., Schultz, M. B., Zobel, J., & Holt, K. E. (2015). "Bandage: interactive visualization of de novo genome assemblies." *Bioinformatics*, 31(20), 3350-3352.

### Minimap2
- Li, H. (2018). "Minimap2: pairwise alignment for nucleotide sequences." *Bioinformatics*, 34(18), 3094-3100.

---

## Data Sources

### Simulated Training Data
- All training data for StrandWeaver AI/ML models generated using:
  - Custom genome simulator (scripts/genome_simulation.py)
  - Technology-specific error models based on published error rates
  - Repeat and SV injection following realistic distributions

### Public Datasets (for testing)
- **Human Reference Genomes**:
  - T2T Consortium. (2022). "The complete sequence of a human genome." *Science*, 376(6588), 44-53.
  - GRCh38 (Genome Reference Consortium Human Build 38)

- **Model Organisms**:
  - *S. cerevisiae*: Saccharomyces Genome Database (SGD)
  - *E. coli*: NCBI RefSeq assemblies

---

## License

All referenced papers are cited for academic purposes. Please consult original publications for full details and licensing information.

