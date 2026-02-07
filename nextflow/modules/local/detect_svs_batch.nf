/*
========================================================================================
    Process: SV Detection Batch (PARALLEL)
========================================================================================
*/

process DETECT_SVS_BATCH {
    label 'process_medium'
    tag "sv_batch_${graph_partition.baseName}"
    
    input:
    path graph_partition
    path ul_routes
    path hic_phasing
    
    output:
    path "svs_${graph_partition.baseName}.vcf", emit: vcf
    
    script:
    def ul_arg = ul_routes ? "--ul-routes ${ul_routes}" : ""
    def hic_arg = hic_phasing ? "--hic-phasing ${hic_phasing}" : ""
    """
    python3 -m strandweaver.cli batch detect-svs \\
        --graph ${graph_partition} \\
        ${ul_arg} \\
        ${hic_arg} \\
        --output svs_${graph_partition.baseName}.vcf \\
        --threads ${task.cpus}
    """
}
