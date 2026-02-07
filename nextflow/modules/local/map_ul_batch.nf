/*
========================================================================================
    Process: UL Read Mapping Batch (PARALLEL)
========================================================================================
*/

process MAP_UL_BATCH {
    label 'process_medium'
    tag "ul_batch_${ul_batch.baseName}"
    
    input:
    path ul_batch
    path graph
    
    output:
    path "ul_mappings_${ul_batch.baseName}.json", emit: mappings
    
    script:
    """
    python3 -m strandweaver.cli batch map-ul \\
        --ul-reads ${ul_batch} \\
        --graph ${graph} \\
        --output ul_mappings_${ul_batch.baseName}.json \\
        --threads ${task.cpus}
    """
}
