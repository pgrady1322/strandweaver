/*
========================================================================================
    Process: PathWeaver General Iteration (SEQUENTIAL - GNN requires full graph)
========================================================================================
*/

process PATHWEAVER_ITER_GENERAL {
    label 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path graph
    
    output:
    path "pathweaver_iter_general.gfa", emit: graph
    path "paths_general.json", emit: paths
    
    script:
    """
    strandweaver nf-pathweaver-iter-general \\
        --graph ${graph} \\
        --output pathweaver_iter_general.gfa \\
        --paths paths_general.json \\
        --enable-ai ${params.enable_ai} \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}
