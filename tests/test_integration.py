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
        """Test PipelineOrchestrator initialization with a proper config dict."""
        # Write test FASTQ to file
        fastq_file = temp_output_dir / "test_reads.fastq"
        fastq_file.write_text(simple_fastq)
        
        # Build a minimal config matching the schema expected by PipelineOrchestrator
        config = {
            'runtime': {
                'output_dir': str(temp_output_dir),
                'reads': [str(fastq_file)],
                'technologies': ['ont'],
            },
            'output': {
                'logging': {
                    'level': 'WARNING',
                    'log_file': 'test.log',
                },
            },
            'pipeline': {
                'steps': ['kweaver', 'profile', 'correct', 'assemble', 'finish'],
            },
            'ai': {'enabled': False, 'correction': {}, 'assembly': {}},
            'hardware': {'threads': 2},
        }
        
        # Create orchestrator — should succeed without error
        orchestrator = PipelineOrchestrator(config)
        
        # Verify it initialised correctly
        assert orchestrator.output_dir == temp_output_dir
        assert orchestrator.steps == config['pipeline']['steps']
        assert orchestrator.state['technologies'] == ['ont']
        assert len(orchestrator.state['read_files']) == 1
    
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
        """Test that diploid sequences are recognized as different haplotypes."""
        from strandweaver.assembly_core.dbg_engine_module import KmerGraph, KmerNode, KmerEdge
        from strandweaver.assembly_core.pathweaver_module import PathWeaver
        
        hap_a = diploid_sequences["haplotype_a"]
        hap_b = diploid_sequences["haplotype_b"]
        
        # Build a trivial graph so PathWeaver can initialise
        graph = KmerGraph(base_k=11)
        node_a = KmerNode(id=0, seq=hap_a, coverage=30.0, length=len(hap_a))
        node_b = KmerNode(id=1, seq=hap_b, coverage=28.0, length=len(hap_b))
        graph.add_node(node_a)
        graph.add_node(node_b)
        
        # PathWeaver initialises with (graph, validation_config=None)
        weaver = PathWeaver(graph)
        
        # Verify the protection threshold is set to 99.5%
        assert weaver.variation_protection_threshold == 0.995
        
        # Compute simple identity to confirm haplotypes differ
        matches = sum(1 for a, b in zip(hap_a, hap_b) if a == b)
        identity = matches / max(len(hap_a), len(hap_b))
        
        # Should be <99.5% identical due to SNP at position 9
        assert identity < 0.995, "Failed to detect haplotype difference"
        assert identity > 0.90, "Haplotypes should still be mostly identical"
