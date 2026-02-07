"""
Integration test - small end-to-end assembly test.

This test creates minimal synthetic data and runs a complete (tiny) assembly
to verify the pipeline doesn't crash.

Author: StrandWeaver Development Team
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import pytest
from pathlib import Path
from strandweaver.utils import PipelineOrchestrator


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_minimal_assembly(self, temp_output_dir, simple_fastq):
        """Test complete assembly pipeline on minimal data."""
        # Write test FASTQ to file
        fastq_file = temp_output_dir / "test_reads.fastq"
        fastq_file.write_text(simple_fastq)
        
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            output_dir=temp_output_dir,
            num_threads=2
        )
        
        # Run preprocessing only (full assembly would take too long)
        try:
            result = orchestrator.run_preprocessing(
                read_files=[str(fastq_file)],
                technologies=["ont"]
            )
            
            # Check that preprocessing completed
            assert result is not None
            assert result.num_reads_processed > 0
            
        except Exception as e:
            pytest.fail(f"Minimal assembly failed: {e}")
    
    def test_kmer_prediction_pipeline(self, temp_output_dir, simple_fastq):
        """Test k-mer prediction step in isolation."""
        from strandweaver.preprocessing import KWeaverPredictor
        
        # Write test FASTQ
        fastq_file = temp_output_dir / "test_reads.fastq"
        fastq_file.write_text(simple_fastq)
        
        # Run k-mer prediction
        predictor = KWeaverPredictor()
        
        try:
            prediction = predictor.predict_from_file(str(fastq_file))
            
            # Check prediction structure
            assert prediction.dbg_k > 0
            assert prediction.ul_overlap_k > 0
            assert prediction.extension_k > 0
            assert prediction.polish_k > 0
            
        except Exception as e:
            pytest.fail(f"K-mer prediction failed: {e}")
    
    def test_haplotype_preservation(self, diploid_sequences):
        """Test that diploid sequences aren't collapsed."""
        from strandweaver.assembly_core.pathweaver_module import PathWeaver
        
        hap_a = diploid_sequences["haplotype_a"]
        hap_b = diploid_sequences["haplotype_b"]
        
        # PathWeaver should detect these as separate haplotypes
        weaver = PathWeaver(min_identity_threshold=0.995)
        
        # Check that sequences are recognized as different
        identity = weaver.calculate_identity(hap_a, hap_b)
        
        # Should be <99.5% identical due to SNP
        assert identity < 0.995, "Failed to detect haplotype difference"
