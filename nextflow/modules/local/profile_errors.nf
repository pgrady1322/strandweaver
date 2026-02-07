/*
========================================================================================
    Process: Error Profiling (SEQUENTIAL)
========================================================================================
*/

process PROFILE_ERRORS {
    label 'process_high_memory'
    publishDir "${params.outdir}/preprocessing", mode: 'copy'
    
    input:
    path hifi_reads
    path ont_reads
    path illumina_r1
    path illumina_r2
    
    output:
    path "error_profiles.json", emit: profiles
    
    script:
    def hifi_arg = hifi_reads.name != 'NO_FILE1' ? "--hifi ${hifi_reads}" : ""
    def ont_arg = ont_reads.name != 'NO_FILE2' ? "--ont ${ont_reads}" : ""
    """
    python3 -m strandweaver.cli batch profile-errors \\
        ${hifi_arg} \\
        ${ont_arg} \\
        --output error_profiles.json \\
        --threads ${task.cpus}
    """
}
