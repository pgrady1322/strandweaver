/*
========================================================================================
    Process: Edge Scoring Batch (PARALLEL)
========================================================================================
*/

process SCORE_EDGES_BATCH {
    label 'process_medium'
    tag "edges_batch_${edge_batch.baseName}"
    
    input:
    path edge_batch
    path alignments
    path error_profiles
    
    output:
    path "edge_scores_${edge_batch.baseName}.json", emit: scores
    
    script:
    """
    strandweaver batch score-edges \\
        --edges ${edge_batch} \\
        --alignments ${alignments} \\
        --profiles ${error_profiles} \\
        --output edge_scores_${edge_batch.baseName}.json \\
        --threads ${task.cpus}
    """
}
