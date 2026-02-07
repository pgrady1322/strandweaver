/*
========================================================================================
    Process: Error Correction Batch (PARALLEL)
========================================================================================
*/

process CORRECT_BATCH {
    label 'process_medium'
    tag "${batch_file.baseName}"
    
    input:
    tuple path(batch_file), val(tech_type), path(error_profiles)
    
    output:
    tuple path("corrected_${batch_file.baseName}.fastq.gz"), val(tech_type), emit: corrected
    
    script:
    """
    python3 -m strandweaver.cli batch correct \\
        --input ${batch_file} \\
        --profiles ${error_profiles} \\
        --technology ${tech_type} \\
        --output corrected_${batch_file.baseName}.fastq.gz \\
        --threads ${task.cpus}
    """
}
