/*
========================================================================================
    Process: PathWeaver Pass A (SEQUENTIAL - GNN requires full graph)
========================================================================================
*/

process PATHWEAVER_PASS_A {
    label 'process_gpu'
    publishDir "${params.outdir}/assembly", mode: 'copy'
    
    input:
    path graph
    
    output:
    path "pathweaver_pass_a.gfa", emit: graph
    path "paths_a.json", emit: paths
    
    script:
    """
    strandweaver pathweaver-pass-a \\
        --graph ${graph} \\
        --output pathweaver_pass_a.gfa \\
        --paths paths_a.json \\
        --enable-ai ${params.enable_ai} \\
        --threads ${task.cpus} \\
        --device ${task.accelerator ? 'cuda' : 'cpu'}
    """
}
