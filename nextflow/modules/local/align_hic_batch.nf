/*
========================================================================================
    Process: Hi-C Alignment Batch (PARALLEL)
========================================================================================
*/

process ALIGN_HIC_BATCH {
    label 'process_medium'
    tag "hic_batch_${hic_batch.baseName}"
    
    input:
    path hic_batch
    path graph
    
    output:
    path "hic_contacts_${hic_batch.baseName}.txt", emit: contacts
    
    script:
    """
    python3 -m strandweaver.cli batch align-hic \\
        --hic-reads ${hic_batch} \\
        --graph ${graph} \\
        --output hic_contacts_${hic_batch.baseName}.txt \\
        --threads ${task.cpus}
    """
}
