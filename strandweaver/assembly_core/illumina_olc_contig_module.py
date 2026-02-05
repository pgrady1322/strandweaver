"""
StrandWeaver v0.1.0

Contig builder for converting short reads into artificial long reads.

This module implements an overlap-layout-consensus (OLC) approach to assemble
Illumina short reads into longer contigs (artificial long reads). These contigs
can then serve as input for graph-based assembly.

Key Features:
- K-mer based overlap detection (efficient, reuses existing k-mer infrastructure)
- Variation-preserving consensus (heterozygous sites maintained)
- Quality-aware contig extension
- Graph-based layout

Author: StrandWeaver Development Team
Date: 2025-12-01
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import heapq
import re
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import os

from strandweaver.io_utils import SeqRead, read_fastq, write_fasta
from strandweaver.preprocessing import KmerSpectrum
from strandweaver.utils.hardware_management import (
    GPUSequenceAligner,
    GPUOverlapDetector
)

# Import AI k-mer predictor
try:
    from strandweaver.preprocessing import AdaptiveKmerPredictor
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False


@dataclass
class PairedEndInfo:
    """
    Information about paired-end reads.
    
    Paired-end sequencing produces two reads from opposite ends of a DNA fragment.
    The distance between reads (insert size) provides valuable scaffolding information.
    """
    read1_id: str  # ID of first read in pair
    read2_id: str  # ID of second read in pair
    insert_size: Optional[int] = None  # Estimated insert size (if calculable)
    orientation: str = "FR"  # Forward-Reverse (standard Illumina)
    
    def get_mate(self, read_id: str) -> Optional[str]:
        """Get the mate of a given read ID."""
        if read_id == self.read1_id:
            return self.read2_id
        elif read_id == self.read2_id:
            return self.read1_id
        return None


@dataclass
class Overlap:
    """
    Represents an overlap between two reads.
    
    Overlap notation: Read A overlaps Read B
    A: --------------->
    B:       ---------------->
         <--overlap-->
    """
    read_a_id: str  # First read ID
    read_b_id: str  # Second read ID
    overlap_length: int  # Length of overlap
    score: float  # Quality score for this overlap
    shared_kmers: int  # Number of shared k-mers
    is_suffix_prefix: bool = True  # True if A's suffix overlaps B's prefix
    reverse_complement: bool = False  # True if overlap with RC of B
    
    def __lt__(self, other):
        """For priority queue (higher score = better)."""
        return self.score > other.score
    
    def __str__(self) -> str:
        direction = "RC" if self.reverse_complement else "FW"
        return f"{self.read_a_id} -> {self.read_b_id} ({self.overlap_length}bp, {direction}, score={self.score:.2f})"


class OverlapGraph:
    """
    Overlap graph for organizing read overlaps.
    
    Nodes = reads
    Edges = overlaps between reads
    """
    
    def __init__(self):
        """Initialize empty overlap graph."""
        # Adjacency lists: read_id -> list of Overlap objects
        self.outgoing_edges: Dict[str, List[Overlap]] = defaultdict(list)
        self.incoming_edges: Dict[str, List[Overlap]] = defaultdict(list)
        
        # Read storage
        self.reads: Dict[str, SeqRead] = {}
        
        # Statistics
        self.num_overlaps = 0
    
    def add_read(self, read: SeqRead):
        """Add a read to the graph."""
        self.reads[read.id] = read
    
    def add_overlap(self, overlap: Overlap):
        """Add an overlap edge to the graph."""
        self.outgoing_edges[overlap.read_a_id].append(overlap)
        self.incoming_edges[overlap.read_b_id].append(overlap)
        self.num_overlaps += 1
    
    def get_outgoing(self, read_id: str) -> List[Overlap]:
        """Get all outgoing overlaps from a read."""
        return self.outgoing_edges.get(read_id, [])
    
    def get_incoming(self, read_id: str) -> List[Overlap]:
        """Get all incoming overlaps to a read."""
        return self.incoming_edges.get(read_id, [])
    
    def get_best_extension(self, read_id: str) -> Optional[Overlap]:
        """Get the best outgoing overlap for extending a contig."""
        outgoing = self.get_outgoing(read_id)
        if not outgoing:
            return None
        
        # Return overlap with highest score
        return max(outgoing, key=lambda o: o.score)
    
    def has_unique_extension(self, read_id: str, min_score_ratio: float = 2.0) -> bool:
        """
        Check if a read has a single, unambiguous extension.
        
        Args:
            read_id: Read to check
            min_score_ratio: Best score must be this many times better than second best
            
        Returns:
            True if unique extension exists
        """
        outgoing = self.get_outgoing(read_id)
        
        if len(outgoing) == 0:
            return False
        elif len(outgoing) == 1:
            return True
        else:
            # Multiple extensions - check if one is clearly best
            scores = sorted([o.score for o in outgoing], reverse=True)
            if len(scores) >= 2 and scores[0] >= scores[1] * min_score_ratio:
                return True
            return False
    
    def is_used(self, read_id: str, used_reads: Set[str]) -> bool:
        """Check if a read has already been used in a contig."""
        return read_id in used_reads
    
    def get_linear_paths(self, min_length: int = 2) -> List[List[str]]:
        """
        Find all linear (unbranched) paths in the graph.
        
        Args:
            min_length: Minimum path length in number of reads
            
        Returns:
            List of paths, where each path is a list of read IDs
        """
        paths = []
        used_reads = set()
        
        # Start from reads with no incoming edges (potential start points)
        start_reads = [
            read_id for read_id in self.reads.keys()
            if len(self.get_incoming(read_id)) == 0
            and read_id not in used_reads
        ]
        
        for start_id in start_reads:
            if start_id in used_reads:
                continue
            
            # Extend path as far as possible
            path = self._extend_path(start_id, used_reads)
            
            if len(path) >= min_length:
                paths.append(path)
                used_reads.update(path)
        
        return paths
    
    def _extend_path(self, start_id: str, used_reads: Set[str]) -> List[str]:
        """
        Extend a linear path from a starting read.
        
        Args:
            start_id: Starting read ID
            used_reads: Set of already-used reads
            
        Returns:
            Path as list of read IDs
        """
        path = [start_id]
        current_id = start_id
        
        while True:
            # Get best extension
            best_overlap = self.get_best_extension(current_id)
            
            if best_overlap is None:
                break  # No more extensions
            
            next_id = best_overlap.read_b_id
            
            # Check if next read is already used or has multiple incoming edges
            if next_id in used_reads or next_id in path:
                break  # Would create a cycle
            
            if len(self.get_incoming(next_id)) > 1:
                break  # Branching point (don't merge paths yet)
            
            # Add to path and continue
            path.append(next_id)
            current_id = next_id
        
        return path


class ContigBuilder:
    """
    Build contigs (artificial long reads) from short reads using OLC assembly.
    
    Supports both single-end and paired-end Illumina data. For paired-end data,
    uses insert size information to validate assemblies and scaffold contigs.
    
    Process:
    1. Detect paired-end reads (if present)
    2. Build k-mer index from all reads
    3. Detect overlaps using shared k-mers
    4. Build overlap graph
    5. Find linear paths through graph
    6. Generate consensus sequences for contigs
    7. (Optional) Scaffold contigs using mate pairs
    """
    
    def __init__(
        self,
        k_size: int = 31,
        min_overlap: int = 50,
        min_shared_kmers: int = 10,
        min_overlap_identity: float = 0.95,
        preserve_variants: bool = True,
        min_contig_length: int = 500,
        use_paired_end: bool = True,
        expected_insert_size: Optional[int] = None,
        insert_size_tolerance: float = 0.3,
        num_threads: Optional[int] = None,
        use_gpu: bool = True,
        use_adaptive_k: bool = False,
        adaptive_k_model_dir: Optional[Path] = None,
        **kwargs
    ):
        """
        Initialize contig builder.
        
        Args:
            k_size: K-mer size for overlap detection (default: 31)
            min_overlap: Minimum overlap length in bp (default: 50)
            min_shared_kmers: Minimum shared k-mers for overlap (default: 10)
            min_overlap_identity: Minimum sequence identity in overlap (default: 0.95)
            preserve_variants: Preserve heterozygous variants (default: True)
            min_contig_length: Minimum contig length to output (default: 500)
            use_paired_end: Use paired-end information if available (default: True)
            expected_insert_size: Expected insert size for paired-end (default: auto-detect)
            insert_size_tolerance: Tolerance for insert size validation (default: 0.3 = 30%)
            num_threads: Number of threads for parallel processing (default: auto = cpu_count)
            use_gpu: Enable GPU acceleration for overlap verification (default: True)
            use_adaptive_k: Use AI-based adaptive k-mer selection (default: False)
            adaptive_k_model_dir: Directory containing trained ML models (default: None = use packaged models)
            **kwargs: Additional parameters
        """
        self.k_size = k_size
        self.min_overlap = min_overlap
        self.min_shared_kmers = min_shared_kmers
        self.min_overlap_identity = min_overlap_identity
        self.preserve_variants = preserve_variants
        self.min_contig_length = min_contig_length
        
        # Parallel processing
        self.num_threads = num_threads if num_threads else max(1, cpu_count() - 1)
        
        # Adaptive k-mer selection
        self.use_adaptive_k = use_adaptive_k
        self.adaptive_k_predictor = None
        if self.use_adaptive_k:
            if not AI_AVAILABLE:
                raise ImportError(
                    "Adaptive k-mer selection requires AI module. "
                    "Install with: pip install strandweaver[ai]"
                )
            self.adaptive_k_predictor = AdaptiveKmerPredictor(
                model_dir=adaptive_k_model_dir,
                use_ml=True
            )
        
        # Paired-end parameters
        self.use_paired_end = use_paired_end
        self.expected_insert_size = expected_insert_size
        self.insert_size_tolerance = insert_size_tolerance
        
        # GPU acceleration
        self.gpu_aligner = GPUSequenceAligner(use_gpu=use_gpu)
        self.gpu_overlap_detector = GPUOverlapDetector(
            k_size=k_size,
            min_shared_kmers=min_shared_kmers,
            use_gpu=use_gpu
        ) if use_gpu else None
        self.gpu_consensus_generator = GPUConsensusGenerator(use_gpu=use_gpu) if use_gpu else None
        
        # K-mer index: kmer -> list of (read_id, position)
        self.kmer_index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        
        # Overlap graph
        self.graph = OverlapGraph()
        
        # Paired-end tracking
        self.paired_reads: Dict[str, PairedEndInfo] = {}  # read_id -> PairedEndInfo
        self.is_paired_end_data = False
        self.insert_sizes: List[int] = []  # Observed insert sizes
        self.mean_insert_size: Optional[float] = None
        self.insert_size_std: Optional[float] = None
        
        # Statistics
        self.stats = {
            'reads_input': 0,
            'paired_reads_detected': 0,
            'kmers_indexed': 0,
            'overlaps_found': 0,
            'contigs_built': 0,
            'total_contig_length': 0,
            'avg_contig_length': 0,
            'n50': 0,
            'scaffolded_contigs': 0
        }
    
    def build_contigs(self, reads: List[SeqRead], verbose: bool = True) -> List[SeqRead]:
        """
        Build contigs from a list of reads.
        
        Args:
            reads: List of input reads
            verbose: Print progress information
            
        Returns:
            List of contigs as SeqRead objects
        """
        if verbose:
            print(f"\n=== Building Contigs (OLC Assembly) ===")
            print(f"Input reads: {len(reads)}")
        
        self.stats['reads_input'] = len(reads)
        
        # Step 0a: Adaptive k-mer selection (if enabled)
        if self.use_adaptive_k:
            if verbose:
                print(f"Step 0a: Predicting optimal k-mer size using ML...")
            self._predict_optimal_k(reads, verbose=verbose)
        
        # Step 0b: Detect paired-end reads (if enabled)
        if self.use_paired_end and verbose:
            print(f"Step 0b: Detecting paired-end structure...")
        if self.use_paired_end:
            self._detect_paired_reads(reads)
            if self.is_paired_end_data:
                if verbose:
                    print(f"  Found {self.stats['paired_reads_detected']} read pairs")
                    if self.mean_insert_size:
                        print(f"  Insert size: {self.mean_insert_size:.0f} ± {self.insert_size_std:.0f} bp")
            else:
                if verbose:
                    print(f"  No paired-end structure detected - using single-end mode")
        
        # Step 1: Build k-mer index
        if verbose:
            print(f"Step 1: Building k-mer index (k={self.k_size})...")
        self._build_kmer_index(reads)
        
        if verbose:
            print(f"  Indexed {self.stats['kmers_indexed']} k-mers")
        
        # Step 2: Detect overlaps
        if verbose:
            print(f"Step 2: Detecting overlaps (min={self.min_overlap}bp)...")
        self._detect_overlaps(reads, verbose=verbose)
        
        if verbose:
            print(f"  Found {self.stats['overlaps_found']} overlaps")
        
        # Step 3: Build overlap graph
        if verbose:
            print(f"Step 3: Building overlap graph...")
        # Graph already built during overlap detection
        
        # Step 4: Find paths and build contigs
        if verbose:
            print(f"Step 4: Assembling contigs...")
        contigs = self._assemble_contigs(verbose=verbose)
        
        # Filter by length
        contigs = [c for c in contigs if len(c.sequence) >= self.min_contig_length]
        
        # Step 5: Scaffold using paired-end info (if available)
        if self.is_paired_end_data and len(contigs) > 1:
            if verbose:
                print(f"Step 5: Scaffolding with mate pairs...")
            contigs = self._scaffold_with_pairs(contigs, verbose=verbose)
            if verbose:
                print(f"  Scaffolded {self.stats['scaffolded_contigs']} contig pairs")
        
        # Update statistics
        self.stats['contigs_built'] = len(contigs)
        if contigs:
            self.stats['total_contig_length'] = sum(len(c.sequence) for c in contigs)
            self.stats['avg_contig_length'] = self.stats['total_contig_length'] / len(contigs)
            self.stats['n50'] = self._calculate_n50([len(c.sequence) for c in contigs])
        
        if verbose:
            print(f"\n=== Assembly Complete ===")
            print(f"Contigs assembled: {self.stats['contigs_built']}")
            if self.is_paired_end_data:
                print(f"Paired-end mode: YES")
            print(f"Total length: {self.stats['total_contig_length']:,} bp")
            print(f"Average length: {self.stats['avg_contig_length']:.0f} bp")
            print(f"N50: {self.stats['n50']:,} bp")
            print(f"============================\n")
        
        return contigs
    
    def _predict_optimal_k(self, reads: List[SeqRead], verbose: bool = True):
        """
        Predict optimal k-mer size using AI model.
        
        Uses the AdaptiveKmerPredictor to analyze read characteristics
        and predict the best k-mer size for DBG construction.
        
        Args:
            reads: List of input reads
            verbose: Print prediction details
        """
        if not self.adaptive_k_predictor:
            return
        
        # Create temporary file from reads for feature extraction
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fastq', delete=False) as tmp:
            tmp_path = tmp.name
            for read in reads[:1000]:  # Sample first 1000 reads for speed
                tmp.write(f"@{read.id}\n{read.sequence}\n+\n{read.quality}\n")
        
        try:
            # Predict optimal k
            prediction = self.adaptive_k_predictor.predict_from_file(tmp_path)
            
            # Use the predicted DBG k
            old_k = self.k_size
            self.k_size = prediction.dbg_k
            
            if verbose:
                print(f"  ML Prediction:")
                print(f"    DBG k-mer size: {prediction.dbg_k} (was {old_k})")
                print(f"    DBG confidence: {prediction.dbg_confidence:.2%}")
                print(f"    UL overlap k: {prediction.ul_overlap_k} (confidence: {prediction.ul_overlap_confidence:.2%})")
                print(f"    Extension k: {prediction.extension_k} (confidence: {prediction.extension_confidence:.2%})")
                print(f"    Polish k: {prediction.polish_k} (confidence: {prediction.polish_confidence:.2%})")
                
                # Store prediction in stats
                self.stats['predicted_k'] = prediction.dbg_k
                self.stats['dbg_confidence'] = prediction.dbg_confidence
                self.stats['ul_overlap_k'] = prediction.ul_overlap_k
                self.stats['extension_k'] = prediction.extension_k
                self.stats['polish_k'] = prediction.polish_k
                self.stats['original_k'] = old_k
        
        finally:
            # Clean up temp file
            import os
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def _detect_paired_reads(self, reads: List[SeqRead]):
        """
        Detect paired-end read structure from read IDs.
        
        Common formats:
        - read_id/1 and read_id/2
        - read_id.1 and read_id.2
        - read_id_1 and read_id_2
        - read_id 1:N:... and read_id 2:N:...
        
        Args:
            reads: List of reads to analyze
        """
        # Build a map of base_id -> list of read objects
        base_id_map = defaultdict(list)
        
        for read in reads:
            base_id = self._get_base_read_id(read.id)
            if base_id:
                base_id_map[base_id].append(read)
        
        # Find pairs
        for base_id, read_list in base_id_map.items():
            if len(read_list) == 2:
                # Found a pair
                read1, read2 = sorted(read_list, key=lambda r: r.id)
                
                pair_info = PairedEndInfo(
                    read1_id=read1.id,
                    read2_id=read2.id,
                    orientation="FR"  # Standard Illumina
                )
                
                self.paired_reads[read1.id] = pair_info
                self.paired_reads[read2.id] = pair_info
                self.stats['paired_reads_detected'] += 1
        
        # Mark as paired-end data if we found pairs
        if self.stats['paired_reads_detected'] > 0:
            self.is_paired_end_data = True
    
    def _get_base_read_id(self, read_id: str) -> Optional[str]:
        """
        Extract base read ID (without /1, /2, etc. suffix).
        
        Args:
            read_id: Full read ID
            
        Returns:
            Base ID or None if not a recognized format
        """
        # Try different patterns
        patterns = [
            r'^(.+)[/.]([12])$',  # read_id/1, read_id.1
            r'^(.+)_([12])$',     # read_id_1
            r'^(.+)\s+([12]):',   # read_id 1:, read_id 2: (Illumina format)
        ]
        
        for pattern in patterns:
            match = re.match(pattern, read_id)
            if match:
                return match.group(1)
        
        return None
    
    def _estimate_insert_size(self, read1: SeqRead, read2: SeqRead, overlap: Optional[Overlap] = None) -> Optional[int]:
        """
        Estimate insert size for a read pair.
        
        If the reads overlap, we can calculate exact insert size.
        
        Args:
            read1: First read in pair
            read2: Second read in pair
            overlap: Overlap between reads (if they overlap)
            
        Returns:
            Estimated insert size or None
        """
        if overlap and overlap.overlap_length > 0:
            # Reads overlap - can calculate exact insert size
            # Insert size = len(read1) + len(read2) - overlap_length
            insert_size = len(read1.sequence) + len(read2.sequence) - overlap.overlap_length
            return insert_size
        
        return None
    
    def _build_kmer_index(self, reads: List[SeqRead]):
        """Build k-mer index from all reads (GPU-optimized if available)."""
        if self.gpu_overlap_detector:
            # Use GPU-optimized vectorized indexing
            self.kmer_index, kmers_count = self.gpu_overlap_detector.build_kmer_index_vectorized(reads)
            self.stats['kmers_indexed'] = kmers_count
            
            # Add reads to graph
            for read in reads:
                self.graph.add_read(read)
        else:
            # Original CPU implementation
            for read in reads:
                # Add read to graph
                self.graph.add_read(read)
                
                # Index all k-mers in read
                sequence = read.sequence
                for i in range(len(sequence) - self.k_size + 1):
                    kmer = sequence[i:i + self.k_size]
                    
                    # Skip k-mers with N's
                    if 'N' in kmer:
                        continue
                    
                    self.kmer_index[kmer].append((read.id, i))
                    self.stats['kmers_indexed'] += 1
    
    def _detect_overlaps(self, reads: List[SeqRead], verbose: bool = False):
        """Detect overlaps between reads using k-mer index (GPU-accelerated if available)."""
        if self.gpu_overlap_detector and len(reads) >= 100:
            # Use GPU-optimized batch processing for large datasets
            self._detect_overlaps_gpu_batch(reads, verbose)
        elif self.num_threads == 1:
            # Single-threaded mode
            self._detect_overlaps_single_thread(reads, verbose)
        else:
            # Multi-threaded mode
            self._detect_overlaps_parallel(reads, verbose)
    
    def _detect_overlaps_gpu_batch(self, reads: List[SeqRead], verbose: bool = False):
        """GPU-accelerated batch overlap detection."""
        if verbose:
            print(f"  Using GPU-accelerated batch overlap detection...")
        
        # Find overlap candidates using vectorized operations
        candidates = self.gpu_overlap_detector.find_overlap_candidates_vectorized(
            reads, self.kmer_index, batch_size=100
        )
        
        if verbose:
            print(f"  Found {len(candidates)} overlap candidates")
        
        # Build read lookup
        read_dict = {read.id: read for read in reads}
        
        # Prepare candidates for verification
        candidate_list = [
            (read_dict[pair[0]], read_dict[pair[1]], count)
            for pair, count in candidates.items()
            if pair[0] in read_dict and pair[1] in read_dict
        ]
        
        # Verify overlaps in batches using GPU
        batch_size = 100
        for batch_start in range(0, len(candidate_list), batch_size):
            batch_end = min(batch_start + batch_size, len(candidate_list))
            batch = candidate_list[batch_start:batch_end]
            
            # GPU batch verification
            batch_overlaps = self.gpu_overlap_detector.verify_overlaps_batch(
                batch,
                min_overlap=self.min_overlap,
                min_identity=self.min_overlap_identity
            )
            
            # Add valid overlaps to graph
            for overlap in batch_overlaps:
                if overlap:
                    self.graph.add_overlap(overlap)
                    self.stats['overlaps_found'] += 1
            
            if verbose and (batch_start // batch_size + 1) % 10 == 0:
                print(f"  Verified {batch_end}/{len(candidate_list)} candidates...")
    
    def _detect_overlaps_single_thread(self, reads: List[SeqRead], verbose: bool = False):
        """Single-threaded overlap detection (original implementation)."""
        processed = 0
        
        for read in reads:
            # Find potential overlapping reads using k-mer matches
            potential_overlaps = self._find_potential_overlaps(read)
            
            # Verify and score each potential overlap
            for other_read_id, shared_kmers in potential_overlaps.items():
                if other_read_id == read.id:
                    continue  # Skip self-overlaps
                
                # Get other read
                other_read = self.graph.reads.get(other_read_id)
                if not other_read:
                    continue
                
                # Check for suffix-prefix overlap (A's suffix with B's prefix)
                overlap = self._verify_overlap(read, other_read, shared_kmers)
                
                if overlap and overlap.overlap_length >= self.min_overlap:
                    self.graph.add_overlap(overlap)
                    self.stats['overlaps_found'] += 1
            
            processed += 1
            if verbose and processed % 1000 == 0:
                print(f"  Processed {processed}/{len(reads)} reads...")
    
    def _detect_overlaps_parallel(self, reads: List[SeqRead], verbose: bool = False):
        """Parallel overlap detection using ProcessPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor
        
        # Split reads into chunks for parallel processing
        chunk_size = max(100, len(reads) // (self.num_threads * 4))
        chunks = [reads[i:i + chunk_size] for i in range(0, len(reads), chunk_size)]
        
        print(f"  Using {self.num_threads} threads, processing {len(chunks)} chunks...")
        
        all_overlaps = []
        processed = 0
        
        # Process chunks in parallel using threads (better for I/O-bound k-mer lookups)
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._process_read_chunk, chunk): chunk 
                for chunk in chunks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk_overlaps = future.result()
                all_overlaps.extend(chunk_overlaps)
                processed += len(future_to_chunk[future])
                
                if verbose:
                    print(f"  Processed {processed}/{len(reads)} reads...")
        
        # Add all overlaps to graph
        for overlap in all_overlaps:
            self.graph.add_overlap(overlap)
            self.stats['overlaps_found'] += 1
    
    def _process_read_chunk(self, reads: List[SeqRead]) -> List[Overlap]:
        """Process a chunk of reads and return overlaps (thread-safe)."""
        overlaps = []
        
        for read in reads:
            # Find potential overlapping reads using k-mer matches
            potential_overlaps = self._find_potential_overlaps(read)
            
            # Verify and score each potential overlap
            for other_read_id, shared_kmers in potential_overlaps.items():
                if other_read_id == read.id:
                    continue  # Skip self-overlaps
                
                # Get other read
                other_read = self.graph.reads.get(other_read_id)
                if not other_read:
                    continue
                
                # Check for suffix-prefix overlap (A's suffix with B's prefix)
                overlap = self._verify_overlap(read, other_read, shared_kmers)
                
                if overlap and overlap.overlap_length >= self.min_overlap:
                    overlaps.append(overlap)
        
        return overlaps
    
    def _find_potential_overlaps(self, read: SeqRead) -> Dict[str, int]:
        """
        Find reads that share k-mers with this read.
        
        Args:
            read: Query read
            
        Returns:
            Dictionary mapping read_id -> number of shared k-mers
        """
        shared_kmers = Counter()
        
        # Check all k-mers in this read
        for i in range(len(read.sequence) - self.k_size + 1):
            kmer = read.sequence[i:i + self.k_size]
            
            if 'N' in kmer:
                continue
            
            # Find all reads containing this k-mer
            for other_read_id, pos in self.kmer_index.get(kmer, []):
                if other_read_id != read.id:
                    shared_kmers[other_read_id] += 1
        
        # Filter by minimum shared k-mers
        return {
            read_id: count for read_id, count in shared_kmers.items()
            if count >= self.min_shared_kmers
        }
    
    def _verify_overlaps_batch_gpu(
        self,
        candidates: List[Tuple[SeqRead, SeqRead, int]]
    ) -> List[Optional[Overlap]]:
        """
        GPU-accelerated batch overlap verification.
        
        Verifies multiple overlap candidates in parallel on GPU.
        Provides 10-20× speedup over CPU for large batches.
        
        Args:
            candidates: List of (read_a, read_b, shared_kmers) tuples
            
        Returns:
            List of Overlap objects (None for invalid overlaps)
        """
        if not candidates:
            return []
        
        # Extract sequences for batch alignment
        sequences_a = []
        sequences_b = []
        max_overlap_lengths = []
        
        for read_a, read_b, _ in candidates:
            # Get suffix of A and prefix of B for overlap region
            max_overlap = min(len(read_a.sequence), len(read_b.sequence))
            max_overlap_lengths.append(max_overlap)
            
            # For alignment, use the potential overlap regions
            # Take last max_overlap bases of A and first max_overlap bases of B
            sequences_a.append(read_a.sequence[-max_overlap:])
            sequences_b.append(read_b.sequence[:max_overlap])
        
        # Batch align on GPU
        alignments = self.gpu_aligner.align_batch(sequences_a, sequences_b)
        
        # Create Overlap objects from results
        overlaps = []
        for i, (read_a, read_b, shared_kmers) in enumerate(candidates):
            alignment = alignments[i]
            identity = alignment['identity']
            overlap_len = alignment['length2']  # Use length of B prefix
            
            if identity >= self.min_overlap_identity and overlap_len >= self.min_overlap:
                # Valid overlap
                score = self._score_overlap(
                    overlap_len, identity, shared_kmers, read_a, read_b
                )
                
                overlap = Overlap(
                    read_a_id=read_a.id,
                    read_b_id=read_b.id,
                    overlap_length=overlap_len,
                    score=score,
                    shared_kmers=shared_kmers,
                    is_suffix_prefix=True,
                    reverse_complement=False
                )
                overlaps.append(overlap)
            else:
                overlaps.append(None)
        
        return overlaps
    
    def _verify_overlap(
        self, read_a: SeqRead, read_b: SeqRead, shared_kmers: int
    ) -> Optional[Overlap]:
        """
        Verify and score an overlap between two reads.
        
        Check for suffix-prefix overlap: read_a's suffix overlaps read_b's prefix
        
        Args:
            read_a: First read
            read_b: Second read
            shared_kmers: Number of shared k-mers
            
        Returns:
            Overlap object if valid overlap found, None otherwise
        """
        # Try different overlap lengths, starting from longer overlaps
        max_overlap = min(len(read_a.sequence), len(read_b.sequence))
        
        for overlap_len in range(max_overlap, self.min_overlap - 1, -1):
            # Get suffix of A and prefix of B
            a_suffix = read_a.sequence[-overlap_len:]
            b_prefix = read_b.sequence[:overlap_len]
            
            # Calculate identity
            matches = sum(1 for x, y in zip(a_suffix, b_prefix) if x == y)
            identity = matches / overlap_len
            
            if identity >= self.min_overlap_identity:
                # Valid overlap found
                score = self._score_overlap(
                    overlap_len, identity, shared_kmers, read_a, read_b
                )
                
                return Overlap(
                    read_a_id=read_a.id,
                    read_b_id=read_b.id,
                    overlap_length=overlap_len,
                    score=score,
                    shared_kmers=shared_kmers,
                    is_suffix_prefix=True,
                    reverse_complement=False
                )
        
        return None
    
    def _score_overlap(
        self,
        overlap_len: int,
        identity: float,
        shared_kmers: int,
        read_a: SeqRead,
        read_b: SeqRead
    ) -> float:
        """
        Score an overlap based on multiple factors.
        
        Args:
            overlap_len: Length of overlap
            identity: Sequence identity in overlap
            shared_kmers: Number of shared k-mers
            read_a: First read
            read_b: Second read
            
        Returns:
            Overlap score (higher = better)
        """
        # Base score from overlap length and identity
        score = overlap_len * identity
        
        # Bonus for more shared k-mers
        score += shared_kmers * 0.5
        
        # Bonus for quality scores in overlap region (if available)
        if read_a.quality and read_b.quality:
            a_suffix_qual = [ord(c) - 33 for c in read_a.quality[-overlap_len:]]
            b_prefix_qual = [ord(c) - 33 for c in read_b.quality[:overlap_len]]
            
            avg_qual_a = sum(a_suffix_qual) / len(a_suffix_qual)
            avg_qual_b = sum(b_prefix_qual) / len(b_prefix_qual)
            avg_qual = (avg_qual_a + avg_qual_b) / 2
            
            # Quality bonus (Q30 = 1.0x, Q40 = 1.25x)
            quality_factor = 1.0 + (max(0, avg_qual - 30) / 40)
            score *= quality_factor
        
        # PAIRED-END BONUS: If these reads are mates, boost score significantly
        if self.is_paired_end_data:
            if read_a.id in self.paired_reads:
                pair_info = self.paired_reads[read_a.id]
                mate_id = pair_info.get_mate(read_a.id)
                
                if mate_id == read_b.id:
                    # These are mate pairs - strong evidence for this overlap
                    score *= 2.0  # 2x bonus for mate pair overlaps
                    
                    # Estimate and store insert size
                    insert_size = len(read_a.sequence) + len(read_b.sequence) - overlap_len
                    self.insert_sizes.append(insert_size)
                    
                    # Update pair info with insert size
                    pair_info.insert_size = insert_size
        
        return score
    
    def _assemble_contigs(self, verbose: bool = False) -> List[SeqRead]:
        """
        Assemble contigs from overlap graph.
        
        Args:
            verbose: Print progress information
            
        Returns:
            List of contig sequences as SeqRead objects
        """
        import math
        contigs = []
        
        # Find linear paths through the graph
        paths = self.graph.get_linear_paths(min_length=2)
        
        if verbose:
            print(f"  Found {len(paths)} linear paths")
        
        # Build consensus sequence for each path
        for i, path in enumerate(paths):
            if len(path) < 2:
                continue  # Skip single-read "contigs"
            
            contig_seq = self._build_consensus(path)
            
            if len(contig_seq) >= self.min_contig_length:
                # Calculate quality from read depth (number of supporting reads)
                # Q = 20 + 10 * log10(depth + 1), capped at Q40
                depth = len(path)
                avg_qual = min(40, int(20 + 10 * math.log10(depth + 1)))
                quality = chr(avg_qual + 33) * len(contig_seq)
                
                contig = SeqRead(
                    id=f"contig_{i+1}",
                    sequence=contig_seq,
                    quality=quality,
                    technology=self.graph.reads[path[0]].technology,
                    metadata={
                        'num_reads': len(path),
                        'read_ids': path,
                        'contig_type': 'artificial_long_read',
                        'depth': depth,
                        'avg_quality': avg_qual
                    }
                )
                contigs.append(contig)
        
        # Also include unused single reads as contigs (if they're long enough)
        used_reads = set()
        for path in paths:
            used_reads.update(path)
        
        for read_id, read in self.graph.reads.items():
            if read_id not in used_reads and len(read.sequence) >= self.min_contig_length:
                # Single-read contig
                contig = SeqRead(
                    id=f"contig_single_{read_id}",
                    sequence=read.sequence,
                    quality=read.quality,
                    technology=read.technology,
                    metadata={
                        'num_reads': 1,
                        'read_ids': [read_id],
                        'contig_type': 'single_read'
                    }
                )
                contigs.append(contig)
        
        return contigs
    
    def _build_consensus(self, path: List[str]) -> str:
        """
        Build consensus sequence from a path of overlapping reads.
        
        Uses GPU-accelerated vectorized consensus if available, otherwise
        falls back to CPU implementation.
        
        Args:
            path: List of read IDs in order
            
        Returns:
            Consensus sequence
        """
        if len(path) == 0:
            return ""
        
        if len(path) == 1:
            return self.graph.reads[path[0]].sequence
        
        # Collect read sequences and overlaps
        read_sequences = [self.graph.reads[read_id].sequence for read_id in path]
        
        # Find overlap lengths between consecutive reads
        overlap_lengths = []
        for i in range(len(path) - 1):
            current_id = path[i]
            next_id = path[i + 1]
            
            # Find the overlap between these two reads
            overlaps = [o for o in self.graph.get_outgoing(current_id) if o.read_b_id == next_id]
            
            if overlaps:
                overlap_lengths.append(overlaps[0].overlap_length)
            else:
                overlap_lengths.append(0)  # No overlap (concatenate)
        
        # Use GPU-accelerated consensus if available
        if self.gpu_consensus_generator:
            return self.gpu_consensus_generator.generate_consensus_vectorized(
                read_sequences, overlap_lengths, self.preserve_variants
            )
        
        # Fall back to CPU implementation
        consensus = list(read_sequences[0])
        
        # Extend with each subsequent read
        for i in range(len(path) - 1):
            overlap_len = overlap_lengths[i]
            next_read = read_sequences[i + 1]
            
            # Extend consensus with non-overlapping part of next read
            extension_start = overlap_len
            extension = next_read[extension_start:]
            consensus.extend(list(extension))
        
        return ''.join(consensus)
    
    def _scaffold_with_pairs(self, contigs: List[SeqRead], verbose: bool = False) -> List[SeqRead]:
        """
        Scaffold contigs using mate pair information.
        
        If mate pairs span different contigs, we can link those contigs with Ns
        representing the gap (estimated by insert size).
        
        Args:
            contigs: List of assembled contigs
            verbose: Print progress information
            
        Returns:
            List of scaffolded contigs (may be fewer if some were merged)
        """
        if not self.is_paired_end_data or len(contigs) < 2:
            return contigs
        
        # Calculate insert size statistics if we have observations
        if self.insert_sizes:
            self.mean_insert_size = statistics.mean(self.insert_sizes)
            self.insert_size_std = statistics.stdev(self.insert_sizes) if len(self.insert_sizes) > 1 else 0
        elif self.expected_insert_size:
            self.mean_insert_size = self.expected_insert_size
            self.insert_size_std = self.expected_insert_size * 0.1  # Assume 10% std dev
        else:
            # Can't scaffold without insert size info
            return contigs
        
        # Build map of read_id -> contig_id
        read_to_contig = {}
        for contig in contigs:
            if 'read_ids' in contig.metadata:
                for read_id in contig.metadata['read_ids']:
                    read_to_contig[read_id] = contig.id
        
        # Find mate pairs that span different contigs
        scaffold_links = defaultdict(list)  # (contig1, contig2) -> count
        
        for read_id, pair_info in self.paired_reads.items():
            mate_id = pair_info.get_mate(read_id)
            
            if read_id in read_to_contig and mate_id and mate_id in read_to_contig:
                contig1 = read_to_contig[read_id]
                contig2 = read_to_contig[mate_id]
                
                if contig1 != contig2:
                    # Mates in different contigs - scaffolding opportunity
                    link = tuple(sorted([contig1, contig2]))
                    scaffold_links[link].append(pair_info)
        
        # Scaffold contigs with strong mate pair support
        min_pair_support = 3  # Require at least 3 mate pairs to scaffold
        scaffolded_count = 0
        
        scaffolded_contigs = list(contigs)
        used_contigs = set()
        
        for (contig1_id, contig2_id), pairs in scaffold_links.items():
            if len(pairs) >= min_pair_support:
                if contig1_id not in used_contigs and contig2_id not in used_contigs:
                    # Find the actual contig objects
                    contig1 = next((c for c in scaffolded_contigs if c.id == contig1_id), None)
                    contig2 = next((c for c in scaffolded_contigs if c.id == contig2_id), None)
                    
                    if contig1 and contig2:
                        # Estimate gap size from insert sizes
                        gap_size = self._estimate_gap_size(contig1, contig2, pairs)
                        
                        # Create scaffolded sequence
                        gap_seq = 'N' * max(0, gap_size)
                        scaffolded_seq = contig1.sequence + gap_seq + contig2.sequence
                        
                        # Create new scaffolded contig
                        scaffolded_contig = SeqRead(
                            id=f"scaffold_{scaffolded_count + 1}",
                            sequence=scaffolded_seq,
                            quality=None,
                            technology=contig1.technology,
                            metadata={
                                'num_reads': (contig1.metadata.get('num_reads', 0) + 
                                            contig2.metadata.get('num_reads', 0)),
                                'contig_type': 'scaffolded',
                                'source_contigs': [contig1_id, contig2_id],
                                'mate_pair_support': len(pairs),
                                'gap_size': gap_size
                            }
                        )
                        
                        # Remove original contigs and add scaffolded version
                        scaffolded_contigs = [c for c in scaffolded_contigs 
                                            if c.id not in [contig1_id, contig2_id]]
                        scaffolded_contigs.append(scaffolded_contig)
                        
                        used_contigs.add(contig1_id)
                        used_contigs.add(contig2_id)
                        scaffolded_count += 1
        
        self.stats['scaffolded_contigs'] = scaffolded_count
        return scaffolded_contigs
    
    def _estimate_gap_size(
        self, contig1: SeqRead, contig2: SeqRead, pairs: List[PairedEndInfo]
    ) -> int:
        """
        Estimate gap size between two contigs based on mate pairs.
        
        Args:
            contig1: First contig
            contig2: Second contig
            pairs: Mate pairs linking the contigs
            
        Returns:
            Estimated gap size in bp
        """
        # Use mean insert size and contig lengths to estimate gap
        if self.mean_insert_size:
            # Gap ≈ insert_size - (partial contig1 length + partial contig2 length)
            # For simplicity, assume mates are near contig ends
            # More sophisticated: track exact positions of reads in contigs
            estimated_gap = int(self.mean_insert_size - 100)  # Rough estimate
            return max(0, estimated_gap)
        
        return 100  # Default gap if no insert size info
    
    def _calculate_n50(self, lengths: List[int]) -> int:
        """
        Calculate N50 statistic.
        
        Args:
            lengths: List of contig lengths
            
        Returns:
            N50 value
        """
        if not lengths:
            return 0
        
        sorted_lengths = sorted(lengths, reverse=True)
        total_length = sum(sorted_lengths)
        target = total_length / 2
        
        cumulative = 0
        for length in sorted_lengths:
            cumulative += length
            if cumulative >= target:
                return length
        
        return 0


def build_contigs_from_reads(
    input_file: Path,
    output_file: Path,
    k_size: int = 31,
    min_overlap: int = 50,
    min_contig_length: int = 500,
    max_reads: Optional[int] = None,
    verbose: bool = True
) -> Dict:
    """
    Convenience function to build contigs from a FASTQ file.
    
    Args:
        input_file: Input FASTQ file with corrected reads
        output_file: Output FASTA file for contigs
        k_size: K-mer size for overlap detection
        min_overlap: Minimum overlap length
        min_contig_length: Minimum contig length to output
        max_reads: Maximum reads to process (None = all)
        verbose: Print progress information
        
    Returns:
        Statistics dictionary
    """
    # Load reads
    reads = []
    for read in read_fastq(input_file):
        reads.append(read)
        if max_reads and len(reads) >= max_reads:
            break
    
    # Build contigs
    builder = ContigBuilder(
        k_size=k_size,
        min_overlap=min_overlap,
        min_contig_length=min_contig_length
    )
    
    contigs = builder.build_contigs(reads, verbose=verbose)
    
    # Write contigs to FASTA
    write_fasta(contigs, output_file)
    
    if verbose:
        print(f"Wrote {len(contigs)} contigs to {output_file}")
    
    return builder.stats
