/*
========================================================================================
    Helper Processes
========================================================================================
*/

process CLASSIFY_READS {
    label 'process_medium'
    publishDir "${params.outdir}/preprocessing", mode: 'copy'
    
    input:
    path reads
    
    output:
    path "read_classification.json", emit: classification
    
    script:
    """
    strandweaver classify \\
        --input ${reads} \\
        --output read_classification.json \\
        --threads ${task.cpus}
    """
}

process MERGE_CORRECTED {
    label 'process_medium'
    publishDir "${params.outdir}/preprocessing", mode: 'copy'
    
    input:
    path hifi_corrected
    path ont_corrected
    
    output:
    path "corrected_reads.fastq.gz", emit: corrected_reads
    
    script:
    def hifi_arg = hifi_corrected ? "-i ${hifi_corrected.join(' -i ')}" : ""
    def ont_arg = ont_corrected ? "-i ${ont_corrected.join(' -i ')}" : ""
    def all_inputs = [hifi_arg, ont_arg].findAll{ it }.join(' ')
    """
    strandweaver batch merge-corrected \\
        ${all_inputs} \\
        --output corrected_reads.fastq.gz
    """
}

process KWEAVER {
    label 'process_high'
    publishDir "${params.outdir}/preprocessing", mode: 'copy'
    
    input:
    path corrected_reads
    
    output:
    path "kmer_predictions.json", emit: kmer_predictions
    path "kweaver_report.txt", emit: report
    
    script:
    def kmer_arg = params.kmer_size ? "--kmer-size ${params.kmer_size}" : ""
    """
    strandweaver kweaver \\
        --input ${corrected_reads} \\
        --output kmer_predictions.json \\
        --report kweaver_report.txt \\
        ${kmer_arg} \\
        --threads ${task.cpus}
    """
}

process EDGEWARDEN_FILTER {
    label 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path graph
    path edge_scores
    
    output:
    path "filtered_graph.gfa", emit: graph
    path "filtering_stats.json", emit: stats
    
    script:
    """
    strandweaver nf-edgewarden-filter \\
        --graph ${graph} \\
        --edge-scores ${edge_scores} \\
        --output filtered_graph.gfa \\
        --stats filtering_stats.json \\
        --enable-ai ${params.enable_ai} \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}

process THREADCOMPASS_AGGREGATE {
    label 'process_high'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path ul_mappings
    path graph
    
    output:
    path "ul_routes.json", emit: routes
    path "ul_evidence.json", emit: evidence
    
    script:
    """
    strandweaver nf-threadcompass-aggregate \\
        --mappings ${ul_mappings} \\
        --graph ${graph} \\
        --output ul_routes.json \\
        --evidence ul_evidence.json \\
        --threads ${task.cpus}
    """
}

process STRANDTETHER_PHASE {
    label 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path hic_contacts
    path graph
    
    output:
    path "contact_matrix.h5", emit: matrix
    path "phasing_info.json", emit: phasing
    path "phasing_stats.json", emit: stats
    
    script:
    """
    strandweaver nf-strandtether-phase \\
        --contacts ${hic_contacts} \\
        --graph ${graph} \\
        --output-matrix contact_matrix.h5 \\
        --output-phasing phasing_info.json \\
        --stats phasing_stats.json \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}

process PATHWEAVER_ITER_STRICT {
    label 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path graph
    path ul_routes
    path hic_phasing
    
    output:
    path "final_graph.gfa", emit: graph
    path "paths_strict.json", emit: paths
    
    script:
    def ul_arg = ul_routes ? "--ul-routes ${ul_routes}" : ""
    def hic_arg = hic_phasing ? "--hic-phasing ${hic_phasing}" : ""
    """
    strandweaver nf-pathweaver-iter-strict \\
        --graph ${graph} \\
        ${ul_arg} \\
        ${hic_arg} \\
        --output final_graph.gfa \\
        --paths paths_strict.json \\
        --enable-ai ${params.enable_ai} \\
        --preserve-heterozygosity ${params.preserve_heterozygosity} \\
        --min-identity ${params.min_identity} \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}

process MERGE_SVS {
    label 'process_low'
    publishDir "${params.outdir}/variants", mode: 'copy'
    
    input:
    path sv_vcfs
    
    output:
    path "structural_variants.vcf", emit: vcf
    path "sv_summary.json", emit: summary
    
    script:
    """
    strandweaver batch merge-svs \\
        --vcfs ${sv_vcfs} \\
        --output structural_variants.vcf \\
        --summary sv_summary.json
    """
}

process EXPORT_ASSEMBLY {
    label 'process_medium'
    publishDir "${params.outdir}/final", mode: 'copy'
    
    input:
    path graph
    path sv_vcf
    
    output:
    path "assembly.fasta", emit: fasta
    path "assembly.gfa", emit: gfa
    path "assembly_stats.json", emit: stats
    path "coverage_*.csv", emit: coverage
    
    script:
    def sv_arg = sv_vcf ? "--svs ${sv_vcf}" : ""
    """
    strandweaver nf-export-assembly \\
        --graph ${graph} \\
        ${sv_arg} \\
        --output-fasta assembly.fasta \\
        --output-gfa assembly.gfa \\
        --stats assembly_stats.json \\
        --export-coverage
    """
}
