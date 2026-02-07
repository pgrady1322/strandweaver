#!/usr/bin/env nextflow
/*
========================================================================================
    StrandWeaver v0.1.0 - Nextflow Pipeline
========================================================================================
    AI-Powered Multi-Technology Genome Assembler
    Github : https://github.com/pgrady1322/strandweaver
----------------------------------------------------------------------------------------
*/

nextflow.enable.dsl = 2

/*
========================================================================================
    VALIDATE & PRINT PARAMETER SUMMARY
========================================================================================
*/

log.info """\
    ===================================
    S T R A N D W E A V E R  v0.1.0
    ===================================
    Input files:
      HiFi       : ${params.hifi ?: 'Not provided'}
      ONT        : ${params.ont ?: 'Not provided'}
      ONT UL     : ${params.ont_ul ?: 'Not provided'}
      Illumina   : ${params.illumina_r1 && params.illumina_r2 ? 'Provided' : 'Not provided'}
      Hi-C       : ${params.hic_r1 && params.hic_r2 ? 'Provided' : 'Not provided'}
    
    Output:
      Directory  : ${params.outdir}
    
    Options:
      Enable AI  : ${params.enable_ai}
      Detect SVs : ${params.detect_svs}
      Huge genome: ${params.huge}
    
    Parallelization:
      Correction batches : ${params.correction_batch_size}
      Edge batches       : ${params.edge_batch_size}
      UL batches         : ${params.ul_batch_size}
      Hi-C batches       : ${params.hic_batch_size}
      SV batches         : ${params.sv_batch_size}
    """
    .stripIndent()

/*
========================================================================================
    NAMED WORKFLOW FOR PIPELINE
========================================================================================
*/

include { STRANDWEAVER } from './workflows/strandweaver'
include { PROFILE_ERRORS } from './modules/local/profile_errors'
include { CORRECT_BATCH } from './modules/local/correct_batch'
include { MERGE_CORRECTED } from './modules/local/classify_reads'
include { EXTRACT_KMERS_BATCH } from './modules/local/extract_kmers_batch'

//
// WORKFLOW: Run main StrandWeaver assembly pipeline
//
workflow {
    // Collect input channels
    ch_hifi = params.hifi ? Channel.fromPath(params.hifi, checkIfExists: true) : Channel.empty()
    ch_ont = params.ont ? Channel.fromPath(params.ont, checkIfExists: true) : Channel.empty()
    ch_ont_ul = params.ont_ul ? Channel.fromPath(params.ont_ul, checkIfExists: true) : Channel.empty()
    ch_illumina_r1 = params.illumina_r1 ? Channel.fromPath(params.illumina_r1, checkIfExists: true) : Channel.empty()
    ch_illumina_r2 = params.illumina_r2 ? Channel.fromPath(params.illumina_r2, checkIfExists: true) : Channel.empty()
    ch_hic_r1 = params.hic_r1 ? Channel.fromPath(params.hic_r1, checkIfExists: true) : Channel.empty()
    ch_hic_r2 = params.hic_r2 ? Channel.fromPath(params.hic_r2, checkIfExists: true) : Channel.empty()
    
    // Run assembly
    STRANDWEAVER(
        ch_hifi,
        ch_ont,
        ch_ont_ul,
        ch_illumina_r1,
        ch_illumina_r2,
        ch_hic_r1,
        ch_hic_r2
    )
}

//
// WORKFLOW: Error correction only
//
workflow CORRECT {
    // Collect input channels
    ch_hifi = params.hifi ? Channel.fromPath(params.hifi, checkIfExists: true) : Channel.value(file('NO_FILE1'))
    ch_ont = params.ont ? Channel.fromPath(params.ont, checkIfExists: true) : Channel.value(file('NO_FILE2'))
    ch_illumina_r1 = params.illumina_r1 ? Channel.fromPath(params.illumina_r1, checkIfExists: true) : Channel.value(file('NO_FILE3'))
    ch_illumina_r2 = params.illumina_r2 ? Channel.fromPath(params.illumina_r2, checkIfExists: true) : Channel.value(file('NO_FILE4'))
    
    // Profile errors (pass all channels, process will handle empty ones)
    PROFILE_ERRORS(
        ch_hifi,
        ch_ont,
        ch_illumina_r1,
        ch_illumina_r2
    )
    
    // Split reads into batches for parallel correction (only for provided files)
    hifi_batches = params.hifi ? 
        Channel.fromPath(params.hifi, checkIfExists: true)
            .splitFastq(by: params.correction_batch_size, file: true)
            .take(params.max_correction_jobs)
            .map { file -> tuple(file, 'hifi') } :
        Channel.empty()
    
    ont_batches = params.ont ?
        Channel.fromPath(params.ont, checkIfExists: true)
            .splitFastq(by: params.correction_batch_size, file: true)
            .take(params.max_correction_jobs)
            .map { file -> tuple(file, 'ont') } :
        Channel.empty()
    
    // Combine all batches and correct in parallel
    all_batches = hifi_batches.mix(ont_batches)
        .combine(PROFILE_ERRORS.out.profiles)
    
    CORRECT_BATCH(all_batches)
    
    // Merge corrected batches by technology
    corrected_hifi = CORRECT_BATCH.out
        .filter { it[1] == 'hifi' }
        .map { it[0] }
        .collect()
        .ifEmpty([])
    
    corrected_ont = CORRECT_BATCH.out
        .filter { it[1] == 'ont' }
        .map { it[0] }
        .collect()
        .ifEmpty([])
    
    MERGE_CORRECTED(
        corrected_hifi,
        corrected_ont
    )
}

//
// WORKFLOW: K-mer extraction (for huge genomes)
//
workflow EXTRACT_KMERS {
    ch_hifi = params.hifi ? Channel.fromPath(params.hifi, checkIfExists: true) : Channel.empty()
    ch_ont = params.ont ? Channel.fromPath(params.ont, checkIfExists: true) : Channel.empty()
    
    // Combine reads
    all_reads = ch_hifi.mix(ch_ont)
    
    // Split into batches
    kmer_batches = all_reads
        .splitFastq(by: params.kmer_batch_size, file: true)
        .take(params.max_kmer_jobs)
    
    EXTRACT_KMERS_BATCH(
        kmer_batches,
        params.kmer_size ?: 31
    )
}

/*
========================================================================================
    THE END
========================================================================================
*/
