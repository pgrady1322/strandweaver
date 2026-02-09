/*
========================================================================================
    Process: Extract K-mers Batch (PARALLEL - for --huge mode)
========================================================================================
*/

process EXTRACT_KMERS_BATCH {
    label 'process_high'
    tag "kmer_batch_${reads_batch.baseName}"
    
    input:
    path reads_batch
    path kmer_predictions
    
    output:
    path "kmers_${reads_batch.baseName}.pkl", emit: kmer_table
    
    script:
    """
    strandweaver batch extract-kmers \\
        --reads ${reads_batch} \\
        --kmer-predictions ${kmer_predictions} \\
        --output kmers_${reads_batch.baseName}.pkl \\
        --threads ${task.cpus}
    """
}
