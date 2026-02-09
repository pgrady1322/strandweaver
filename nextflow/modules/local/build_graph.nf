/*
========================================================================================
    Process: Build Graph (SEQUENTIAL or uses batched k-mers for --huge)
========================================================================================
*/

process BUILD_GRAPH {
    label params.huge ? 'process_high' : 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path reads_or_kmers  // Either corrected reads or k-mer tables
    path kmer_predictions
    val huge_mode
    
    output:
    path "assembly_graph.gfa", emit: graph
    path "alignments.bam", emit: alignments
    path "build_stats.json", emit: stats
    
    script:
    def input_arg = huge_mode ? "--kmer-tables ${reads_or_kmers}" : "--reads ${reads_or_kmers}"
    """
    strandweaver nf-build-graph \\
        ${input_arg} \\
        --kmer-predictions ${kmer_predictions} \\
        --output assembly_graph.gfa \\
        --alignments alignments.bam \\
        --stats build_stats.json \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}
