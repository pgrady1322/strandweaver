/*
========================================================================================
    StrandWeaver v0.1.0 - Main Assembly Workflow
========================================================================================
*/

include { CLASSIFY_READS } from '../modules/local/classify_reads'
include { PROFILE_ERRORS } from '../modules/local/profile_errors'
include { CORRECT_BATCH } from '../modules/local/correct_batch'
include { MERGE_CORRECTED } from '../modules/local/classify_reads'
include { KWEAVER } from '../modules/local/classify_reads'
include { EXTRACT_KMERS_BATCH } from '../modules/local/extract_kmers_batch'
include { BUILD_GRAPH } from '../modules/local/build_graph'
include { SCORE_EDGES_BATCH } from '../modules/local/score_edges_batch'
include { EDGEWARDEN_FILTER } from '../modules/local/classify_reads'
include { PATHWEAVER_PASS_A } from '../modules/local/pathweaver_pass_a'
include { MAP_UL_BATCH } from '../modules/local/map_ul_batch'
include { THREADCOMPASS_AGGREGATE } from '../modules/local/classify_reads'
include { ALIGN_HIC_BATCH } from '../modules/local/align_hic_batch'
include { STRANDTETHER_PHASE } from '../modules/local/classify_reads'
include { PATHWEAVER_PASS_B } from '../modules/local/classify_reads'
include { DETECT_SVS_BATCH } from '../modules/local/detect_svs_batch'
include { MERGE_SVS } from '../modules/local/classify_reads'
include { EXPORT_ASSEMBLY } from '../modules/local/classify_reads'

workflow STRANDWEAVER {
    take:
        hifi_reads
        ont_reads
        ont_ul_reads
        illumina_r1
        illumina_r2
        hic_r1
        hic_r2
    
    main:
        // ============== SEQUENTIAL: CLASSIFICATION ==============
        all_long_reads = hifi_reads.mix(ont_reads).mix(ont_ul_reads)
        
        CLASSIFY_READS(
            all_long_reads.collect()
        )
        
        // ============== SEQUENTIAL: ERROR PROFILING ==============
        // Profile ALL reads at once for accurate error model
        PROFILE_ERRORS(
            hifi_reads.collect(),
            ont_reads.collect(),
            illumina_r1.collect(),
            illumina_r2.collect()
        )
        
        // ============== PARALLEL: ERROR CORRECTION ==============
        // Split reads into batches for parallel correction
        hifi_batches = hifi_reads
            .splitFastq(by: params.correction_batch_size, file: true)
            .take(params.max_correction_jobs)
            .map { file -> tuple(file, 'hifi') }
        
        ont_batches = ont_reads
            .splitFastq(by: params.correction_batch_size, file: true)
            .take(params.max_correction_jobs)
            .map { file -> tuple(file, 'ont') }
        
        // Combine all batches and correct in parallel
        all_batches = hifi_batches.mix(ont_batches)
            .combine(PROFILE_ERRORS.out.profiles)
        
        CORRECT_BATCH(all_batches)
        
        // Merge corrected batches
        corrected_hifi = CORRECT_BATCH.out
            .filter { it[1] == 'hifi' }
            .map { it[0] }
            .collect()
        
        corrected_ont = CORRECT_BATCH.out
            .filter { it[1] == 'ont' }
            .map { it[0] }
            .collect()
        
        MERGE_CORRECTED(
            corrected_hifi,
            corrected_ont
        )
        
        // ============== SEQUENTIAL: K-WEAVER ==============
        KWEAVER(
            MERGE_CORRECTED.out.corrected_reads
        )
        
        // ============== GRAPH BUILDING (Sequential or Parallel) ==============
        if (params.huge) {
            // PARALLEL: For huge genomes, extract k-mers in batches
            kmer_batches = MERGE_CORRECTED.out.corrected_reads
                .splitFastq(by: params.kmer_batch_size, file: true)
                .take(params.max_kmer_jobs)
            
            EXTRACT_KMERS_BATCH(
                kmer_batches,
                KWEAVER.out.kmer_predictions
            )
            
            // Build graph from merged k-mer tables
            BUILD_GRAPH(
                EXTRACT_KMERS_BATCH.out.collect(),
                KWEAVER.out.kmer_predictions,
                true  // huge mode flag
            )
        } else {
            // SEQUENTIAL: Normal graph building (GPU optimized)
            BUILD_GRAPH(
                MERGE_CORRECTED.out.corrected_reads,
                KWEAVER.out.kmer_predictions,
                false  // not huge mode
            )
        }
        
        // ============== PARALLEL: EDGE SCORING ==============
        // Extract edges from graph and split into batches
        edge_batches = BUILD_GRAPH.out.graph
            .map { graph -> extractEdges(graph, params.edge_batch_size) }
            .flatten()
            .take(params.max_edge_jobs)
        
        SCORE_EDGES_BATCH(
            edge_batches,
            BUILD_GRAPH.out.alignments,
            PROFILE_ERRORS.out.profiles
        )
        
        // ============== SEQUENTIAL: EDGEWARDEN FILTERING ==============
        EDGEWARDEN_FILTER(
            BUILD_GRAPH.out.graph,
            SCORE_EDGES_BATCH.out.collect()
        )
        
        // ============== SEQUENTIAL: PATHWEAVER PASS A ==============
        PATHWEAVER_PASS_A(
            EDGEWARDEN_FILTER.out.graph
        )
        
        // ============== PARALLEL: UL READ MAPPING ==============
        ul_routes = Channel.empty()
        
        if (ont_ul_reads) {
            ul_batches = ont_ul_reads
                .splitFastq(by: params.ul_batch_size, file: true)
                .take(params.max_ul_jobs)
            
            MAP_UL_BATCH(
                ul_batches,
                PATHWEAVER_PASS_A.out.graph
            )
            
            // Aggregate UL mappings
            THREADCOMPASS_AGGREGATE(
                MAP_UL_BATCH.out.collect(),
                PATHWEAVER_PASS_A.out.graph
            )
            
            ul_routes = THREADCOMPASS_AGGREGATE.out.routes
        }
        
        // ============== PARALLEL: HI-C PROCESSING ==============
        hic_phasing = Channel.empty()
        
        if (hic_r1 && hic_r2) {
            // Split Hi-C reads into batches for parallel alignment
            hic_batches = hic_r1.mix(hic_r2)
                .splitFastq(by: params.hic_batch_size, file: true, pe: true)
                .take(params.max_hic_jobs)
            
            ALIGN_HIC_BATCH(
                hic_batches,
                PATHWEAVER_PASS_A.out.graph
            )
            
            // Build contact matrix and phase
            STRANDTETHER_PHASE(
                ALIGN_HIC_BATCH.out.collect(),
                PATHWEAVER_PASS_A.out.graph
            )
            
            hic_phasing = STRANDTETHER_PHASE.out.phasing
        }
        
        // ============== SEQUENTIAL: PATHWEAVER PASS B ==============
        PATHWEAVER_PASS_B(
            PATHWEAVER_PASS_A.out.graph,
            ul_routes.ifEmpty([]),
            hic_phasing.ifEmpty([])
        )
        
        // ============== PARALLEL: SV DETECTION ==============
        sv_vcf = Channel.empty()
        
        if (params.detect_svs) {
            // Partition graph into regions for parallel SV detection
            sv_batches = PATHWEAVER_PASS_B.out.graph
                .map { graph -> partitionGraph(graph, params.sv_batch_size) }
                .flatten()
                .take(params.max_sv_jobs)
            
            DETECT_SVS_BATCH(
                sv_batches,
                ul_routes.ifEmpty([]),
                hic_phasing.ifEmpty([])
            )
            
            MERGE_SVS(
                DETECT_SVS_BATCH.out.collect()
            )
            
            sv_vcf = MERGE_SVS.out.vcf
        }
        
        // ============== SEQUENTIAL: EXPORT ==============
        EXPORT_ASSEMBLY(
            PATHWEAVER_PASS_B.out.graph,
            sv_vcf.ifEmpty([])
        )
    
    emit:
        assembly = EXPORT_ASSEMBLY.out.fasta
        graph = EXPORT_ASSEMBLY.out.gfa
        stats = EXPORT_ASSEMBLY.out.stats
        svs = sv_vcf
}

// Helper function to extract edges from graph into batches
def extractEdges(graph_file, batch_size) {
    // This would be implemented in a helper script
    // Returns list of edge batch files
    def edges = []
    // Parse GFA and split edges
    return edges
}

// Helper function to partition graph by nodes
def partitionGraph(graph_file, nodes_per_partition) {
    // This would be implemented in a helper script
    // Returns list of graph partition files
    def partitions = []
    // Parse GFA and create partitions
    return partitions
}
