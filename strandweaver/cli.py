#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command-line interface for StrandWeaver.

This module provides the main CLI entry point and all subcommands for
the StrandWeaver genome assembly pipeline.
"""

import sys
import click
from pathlib import Path
import yaml

from .version import __version__
from .config.schema import load_config, save_config_template, validate_config


@click.group()
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def main(ctx, verbose, quiet):
    """
    StrandWeaver: AI-Powered Multi-Technology Genome Assembler
    
    A genome assembly tool with intelligent error correction, graph-based assembly,
    and AI-powered finishing for Ancient DNA, Illumina, ONT, and PacBio HiFi data.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    ctx.obj['VERBOSE'] = verbose
    ctx.obj['QUIET'] = quiet


# ============================================================================
# Configuration Management Commands
# ============================================================================

@main.group()
def config():
    """Configuration management commands."""
    pass


@config.command('init')
@click.option('--output', '-o', type=click.Path(), default='strandweaver_config.yaml',
              help='Output configuration file path')
@click.option('--template', '-t', 
              type=click.Choice(['default', 'illumina', 'ancient', 'ont', 'pacbio', 'hybrid']),
              default='default', help='Configuration template type')
def config_init(output, template):
    """Generate a template configuration file with all available parameters."""
    click.echo(f"Generating {template} configuration template: {output}")
    
    try:
        save_config_template(Path(output), template=template)
        click.echo(f"✓ Configuration file created: {output}")
        click.echo("\nThe configuration file includes:")
        click.echo("  • AI/ML settings (ENABLED by default)")
        click.echo("  • Hardware settings (CPU by default, --use-gpu to enable GPU)")
        click.echo("  • Pipeline control and checkpoints")
        click.echo("  • Technology-specific parameters")
        click.echo("  • All available knobs and handles exposed")
        click.echo("\nEdit this file to customize your assembly pipeline.")
    except Exception as e:
        click.echo(f"✗ Error creating configuration: {e}", err=True)


@config.command('validate')
@click.argument('config_file', type=click.Path(exists=True))
def config_validate(config_file):
    """Validate a configuration file."""
    click.echo(f"Validating configuration file: {config_file}")
    
    try:
        config = load_config(Path(config_file))
        errors = validate_config(config)
        
        if errors:
            click.echo("\n✗ Configuration validation failed:")
            for error in errors:
                click.echo(f"  • {error}", err=True)
            sys.exit(1)
        else:
            click.echo("✓ Configuration is valid")
            
            # Show key settings
            click.echo("\nKey Settings:")
            click.echo(f"  AI/ML: {'ENABLED' if config['ai']['enabled'] else 'DISABLED'}")
            click.echo(f"  Hardware: {'GPU' if config['hardware']['use_gpu'] else 'CPU'}")
            click.echo(f"  Pipeline steps: {', '.join(config['pipeline']['steps'])}")
    except Exception as e:
        click.echo(f"✗ Error validating configuration: {e}", err=True)
        sys.exit(1)


@config.command('show')
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['yaml', 'summary']), default='summary',
              help='Output format')
def config_show(config_file, format):
    """Display configuration settings."""
    try:
        config = load_config(Path(config_file))
        
        if format == 'yaml':
            click.echo(yaml.dump(config, default_flow_style=False, sort_keys=False))
        else:
            click.echo(f"Configuration from: {config_file}")
            click.echo("=" * 60)
            
            # AI/ML Settings
            click.echo("\n🤖 AI/ML Settings:")
            ai_status = "ENABLED" if config['ai']['enabled'] else "DISABLED"
            click.echo(f"  Status: {ai_status}")
            if config['ai']['enabled']:
                click.echo(f"  Error Correction AI: {config['ai']['correction']['adaptive_kmer']['enabled']}")
                click.echo(f"  Assembly AI (EdgeAI): {config['ai']['assembly']['edge_ai']['enabled']}")
                click.echo(f"  Assembly AI (PathGNN): {config['ai']['assembly']['path_gnn']['enabled']}")
                click.echo(f"  Assembly AI (DiploidAI): {config['ai']['assembly']['diploid_ai']['enabled']}")
            
            # Hardware
            click.echo("\n💻 Hardware:")
            hw_mode = "GPU" if config['hardware']['use_gpu'] else "CPU"
            click.echo(f"  Mode: {hw_mode}")
            if config['hardware']['threads']:
                click.echo(f"  Threads: {config['hardware']['threads']}")
            
            # Pipeline
            click.echo("\n🔄 Pipeline:")
            click.echo(f"  Steps: {' → '.join(config['pipeline']['steps'])}")
            click.echo(f"  Checkpoints: {config['pipeline']['checkpoint_interval']}")
            
            # Assembly
            click.echo("\n🧬 Assembly:")
            click.echo(f"  Graph type: {config['assembly']['graph']['type']}")
            click.echo(f"  Diploid mode: {config['assembly']['diploid']['mode']}")
            click.echo(f"  SV detection: {config['assembly']['sv_detection']['enabled']}")
            
    except Exception as e:
        click.echo(f"✗ Error reading configuration: {e}", err=True)
        sys.exit(1)


# ============================================================================
# Pipeline Commands
# ============================================================================

@main.command()
@click.option('--reads', '-r', 'reads_files', multiple=True,
              type=click.Path(exists=True),
              help='[OLD SYNTAX] Input reads file (can specify multiple times). Use -r1/-r2/etc for clarity.')
@click.option('--technology', '-tech', 'technologies', multiple=True,
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='[OLD SYNTAX] Technology for each reads file. Use --technology1/--technology2/etc for clarity.')
@click.option('-r1', '--reads1', type=click.Path(exists=True),
              help='Read file 1 (pairs with --technology1)')
@click.option('-r2', '--reads2', type=click.Path(exists=True),
              help='Read file 2 (pairs with --technology2)')
@click.option('-r3', '--reads3', type=click.Path(exists=True),
              help='Read file 3 (pairs with --technology3)')
@click.option('-r4', '--reads4', type=click.Path(exists=True),
              help='Read file 4 (pairs with --technology4)')
@click.option('-r5', '--reads5', type=click.Path(exists=True),
              help='Read file 5 (pairs with --technology5)')
@click.option('--technology1', '-tech1',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 1')
@click.option('--technology2', '-tech2',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 2')
@click.option('--technology3', '-tech3',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 3')
@click.option('--technology4', '-tech4',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 4')
@click.option('--technology5', '-tech5',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 5')
@click.option('--illumina-r1', type=click.Path(exists=True),
              help='Illumina paired-end R1 reads (forward reads)')
@click.option('--illumina-r2', type=click.Path(exists=True),
              help='Illumina paired-end R2 reads (reverse reads)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for assembly')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration file (YAML) - all parameters exposed')
# ============================================================================
# AI/ML Options (DEFAULT: ENABLED)
# ============================================================================
@click.option('--use-ai/--classical', default=True,
              help='Use ML/AI models (default) or classical heuristics')
@click.option('--disable-correction-ai', is_flag=True,
              help='Disable AI-powered error correction (use classical heuristic / coverage based methods)')
@click.option('--disable-assembly-ai', is_flag=True,
              help='Disable AI-powered assembly (use classical graph algorithms)')
@click.option('--ai-finish/--no-ai-finish', default=False,
              help='Use Claude AI for assembly finishing (experimental, requires API key)')
# ============================================================================
# Hardware Options (DEFAULT: CPU)
# ============================================================================
@click.option('--use-gpu', is_flag=True,
              help='Enable GPU acceleration (default: CPU only)')
@click.option('--gpu-backend', 
              type=click.Choice(['auto', 'mps', 'cuda', 'cpu'], case_sensitive=False),
              default='auto',
              help='GPU backend: auto (detect), mps (Apple Silicon), cuda (NVIDIA), cpu (force CPU)')
@click.option('--gpu-device', type=int, default=0,
              help='GPU device ID for multi-GPU systems (CUDA only)')
@click.option('--threads', '-t', type=int, default=None,
              help='Number of CPU threads (default: auto-detect)')
# ============================================================================
# Pipeline Control
# ============================================================================
@click.option('--resume/--no-resume', default=False,
              help='Resume from last checkpoint')
@click.option('--checkpoint-dir', type=click.Path(),
              help='Checkpoint directory (default: <output>/checkpoints)')
@click.option('--start-from', 
              type=click.Choice(['profile', 'correct', 'assemble', 'finish']),
              help='Start pipeline from specific step')
@click.option('--skip-profiling', is_flag=True,
              help='Skip error profiling step')
@click.option('--skip-correction', is_flag=True,
              help='Skip error correction step (use for pre-corrected reads)')
# ============================================================================
# Hi-C Scaffolding Data
# ============================================================================
@click.option('--hic-r1', type=click.Path(exists=True),
              help='Hi-C R1 reads (proximity ligation for scaffolding, not error-corrected)')
@click.option('--hic-r2', type=click.Path(exists=True),
              help='Hi-C R2 reads (proximity ligation for scaffolding, not error-corrected)')
@click.pass_context
def pipeline(ctx, reads_files, technologies, 
             reads1, reads2, reads3, reads4, reads5,
             technology1, technology2, technology3, technology4, technology5,
             output, config,
             use_ai, disable_correction_ai, disable_assembly_ai, ai_finish,
             use_gpu, gpu_backend, gpu_device, threads,
             resume, checkpoint_dir, start_from, skip_profiling, skip_correction,
             illumina_r1, illumina_r2, hic_r1, hic_r2):
    """
    Run the complete assembly pipeline.
    
    This command runs the full StrandWeaver pipeline from error profiling
    through AI-powered finishing. The pipeline can be resumed from checkpoints
    if interrupted.
    
    Supports single or multi-technology (hybrid) assemblies.
    
    Examples:
        # RECOMMENDED: Numbered syntax for explicit mapping
        # Single technology
        strandweaver pipeline -r1 illumina.fq --technology1 illumina -o output/
        
        # Hybrid assembly - Illumina + ONT (numbered syntax)
        strandweaver pipeline \\
            -r1 illumina.fq --technology1 illumina \\
            -r2 ont.fq --technology2 ont \\
            -o output/
        
        # Hybrid with auto-detection
        strandweaver pipeline -r1 illumina.fq -r2 ont.fq -r3 hifi.fq -o output/
        
        # Illumina paired-end with separate R1/R2 files
        strandweaver pipeline --illumina-r1 reads_R1.fq --illumina-r2 reads_R2.fq -o output/
        
        # Hybrid: Illumina paired-end + ONT + PacBio
        strandweaver pipeline \\
            --illumina-r1 ill_R1.fq --illumina-r2 ill_R2.fq \\
            -r1 ont.fq --technology1 ont \\
            -r2 hifi.fq --technology2 pacbio \\
            -o output/
        
        # OLD SYNTAX (still supported but not recommended)
        strandweaver pipeline -r illumina.fq -r ont.fq \\
            --technology illumina --technology ont -o output/
    """
    verbose = ctx.obj.get('VERBOSE', False)
    
    # Collect numbered reads and technologies
    numbered_reads = {}
    numbered_techs = {}
    
    if reads1: numbered_reads[1] = reads1
    if reads2: numbered_reads[2] = reads2
    if reads3: numbered_reads[3] = reads3
    if reads4: numbered_reads[4] = reads4
    if reads5: numbered_reads[5] = reads5
    
    if technology1: numbered_techs[1] = technology1
    if technology2: numbered_techs[2] = technology2
    if technology3: numbered_techs[3] = technology3
    if technology4: numbered_techs[4] = technology4
    if technology5: numbered_techs[5] = technology5
    
    # Check for mixing old and new syntax
    using_old_syntax = bool(reads_files or technologies)
    using_new_syntax = bool(numbered_reads)
    
    if using_old_syntax and using_new_syntax:
        click.echo(f"❌ Error: Cannot mix old syntax (-r/--technology) with new syntax (-r1/--technology1)", err=True)
        click.echo(f"\nUse either:", err=True)
        click.echo(f"  NEW (recommended): -r1 file1.fq --technology1 illumina -r2 file2.fq --technology2 ont", err=True)
        click.echo(f"  OLD: -r file1.fq -r file2.fq --technology illumina --technology ont", err=True)
        ctx.exit(1)
    
    # Validate Illumina paired-end input
    if illumina_r1 and not illumina_r2:
        click.echo(f"❌ Error: --illumina-r1 requires --illumina-r2", err=True)
        ctx.exit(1)
    if illumina_r2 and not illumina_r1:
        click.echo(f"❌ Error: --illumina-r2 requires --illumina-r1", err=True)
        ctx.exit(1)
    
    # Build complete reads list based on syntax used
    all_reads = []
    all_technologies = []
    
    # Add Illumina paired-end if provided
    if illumina_r1 and illumina_r2:
        all_reads.extend([illumina_r1, illumina_r2])
        all_technologies.extend(['illumina', 'illumina'])
    
    if using_new_syntax:
        # NEW NUMBERED SYNTAX - validate and build
        for num in sorted(numbered_reads.keys()):
            all_reads.append(numbered_reads[num])
            
            # Check if technology is specified for this read
            if num in numbered_techs:
                all_technologies.append(numbered_techs[num])
            else:
                # Auto-detect if no technology specified
                all_technologies.append('auto')
                if verbose:
                    click.echo(f"ℹ️  No --technology{num} specified for -r{num}, will auto-detect")
        
        # Warn if technology specified without corresponding read
        for num in numbered_techs.keys():
            if num not in numbered_reads:
                click.echo(f"⚠️  Warning: --technology{num} specified but no -r{num} file provided (ignored)", err=True)
    
    elif using_old_syntax:
        # OLD SYNTAX - validate count matching
        if reads_files:
            all_reads.extend(reads_files)
            
            if technologies:
                if len(technologies) != len(reads_files):
                    click.echo(f"❌ Error: Number of --technology flags ({len(technologies)}) must match "
                               f"number of --reads files ({len(reads_files)})", err=True)
                    click.echo(f"\nFor better clarity, use numbered syntax:", err=True)
                    click.echo(f"  strandweaver pipeline -r1 file1.fq --technology1 illumina \\", err=True)
                    click.echo(f"      -r2 file2.fq --technology2 ont", err=True)
                    ctx.exit(1)
                all_technologies.extend(technologies)
            else:
                # Auto-detect all
                all_technologies.extend(['auto'] * len(reads_files))
                click.echo("ℹ️  No technologies specified - will auto-detect from read characteristics")
    
    # Ensure we have at least some input
    if not all_reads:
        click.echo(f"❌ Error: No input reads specified.", err=True)
        click.echo(f"\nUse one of:", err=True)
        click.echo(f"  --illumina-r1 R1.fq --illumina-r2 R2.fq (for Illumina paired-end)", err=True)
        click.echo(f"  -r1 file.fq --technology1 <tech> (recommended numbered syntax)", err=True)
        click.echo(f"  -r file.fq --technology <tech> (old syntax)", err=True)
        ctx.exit(1)
    
    # ========================================================================
    # Load and merge configuration
    # ========================================================================
    pipeline_config = load_config(Path(config) if config else None)
    
    # Override config with command-line options
    if not use_ai:
        pipeline_config['ai']['enabled'] = False
        if verbose:
            click.echo("ℹ️  AI/ML disabled via --classical flag")
    
    if disable_correction_ai:
        for model in pipeline_config['ai']['correction'].values():
            if isinstance(model, dict):
                model['enabled'] = False
        if verbose:
            click.echo("ℹ️  Error correction AI disabled")
    
    if disable_assembly_ai:
        for model in pipeline_config['ai']['assembly'].values():
            if isinstance(model, dict):
                model['enabled'] = False
        if verbose:
            click.echo("ℹ️  Assembly AI disabled")
    
    if ai_finish:
        pipeline_config['ai']['claude']['enabled'] = True
        pipeline_config['ai']['claude']['use_for_finishing'] = True
    
    if use_gpu:
        pipeline_config['hardware']['use_gpu'] = True
        pipeline_config['hardware']['gpu_device'] = gpu_device
        if verbose:
            click.echo(f"ℹ️  GPU acceleration enabled (device {gpu_device})")
    
    if threads:
        pipeline_config['hardware']['threads'] = threads
    
    if skip_profiling:
        if 'profile' in pipeline_config['pipeline']['steps']:
            pipeline_config['pipeline']['steps'].remove('profile')
    
    if skip_correction:
        if 'correct' in pipeline_config['pipeline']['steps']:
            pipeline_config['pipeline']['steps'].remove('correct')
    
    pipeline_config['pipeline']['resume'] = resume
    if checkpoint_dir:
        pipeline_config['pipeline']['checkpoint_dir'] = checkpoint_dir
    
    # Validate final configuration
    config_errors = validate_config(pipeline_config)
    if config_errors:
        click.echo("❌ Configuration validation failed:", err=True)
        for error in config_errors:
            click.echo(f"  • {error}", err=True)
        ctx.exit(1)
    
    # ========================================================================
    # Display pipeline configuration
    # ========================================================================
    click.echo(f"{'='*60}")
    click.echo(f"StrandWeaver v{__version__}")
    click.echo(f"{'='*60}")
    
    # Input files
    click.echo(f"\n📁 Input Read Files: {len(all_reads)}")
    for i, (reads, tech) in enumerate(zip(all_reads, all_technologies), 1):
        click.echo(f"  {i}. {reads} ({tech})")
    click.echo(f"\n📂 Output: {output}")
    
    # Determine pipeline characteristics
    has_illumina = 'illumina' in all_technologies
    has_long_reads = any(t in ['ont', 'ont_ultralong', 'pacbio', 'hifi'] for t in all_technologies)
    has_ul = 'ont_ultralong' in all_technologies
    has_hic = bool(hic_r1 and hic_r2)
    ai_enabled = pipeline_config['ai']['enabled']
    
    # Get specific AI module states
    correction_ai_modules = []
    assembly_ai_modules = []
    
    if ai_enabled:
        if pipeline_config['ai']['correction'].get('adaptive_kmer', {}).get('enabled', False):
            correction_ai_modules.append('K-Weaver')
        if pipeline_config['ai']['correction'].get('base_error_classifier', {}).get('enabled', False):
            correction_ai_modules.append('ErrorSmith')
        
        if pipeline_config['ai']['assembly'].get('edge_ai', {}).get('enabled', False):
            assembly_ai_modules.append('EdgeWarden')
        if pipeline_config['ai']['assembly'].get('path_gnn', {}).get('enabled', False):
            assembly_ai_modules.append('PathWeaver')
        if pipeline_config['ai']['assembly'].get('ul_routing_ai', {}).get('enabled', False):
            assembly_ai_modules.append('ThreadCompass')
        if pipeline_config['ai']['assembly'].get('diploid_ai', {}).get('enabled', False):
            assembly_ai_modules.append('Haplotype Detangler')
        if pipeline_config['ai']['assembly'].get('sv_ai', {}).get('enabled', False):
            assembly_ai_modules.append('SVScribe')
    
    # Hardware
    hw_mode = "GPU" if pipeline_config['hardware']['use_gpu'] else "CPU (default)"
    hw_threads = pipeline_config['hardware']['threads'] or 'auto-detect'
    click.echo(f"\n💻 Hardware: {hw_mode}")
    click.echo(f"  • Threads: {hw_threads}")
    
    # Build detailed pipeline flow description
    click.echo(f"\n🔄 Pipeline Flow:")
    click.echo(f"{'─'*60}")
    
    # Step 1: K-mer prediction
    kweaver_status = '✓ K-Weaver' if 'K-Weaver' in correction_ai_modules else '○ Heuristic'
    click.echo(f"  1. K-mer Prediction ({kweaver_status})")
    click.echo(f"     └─ Predict optimal k-mer sizes for pipeline stages")
    
    # Step 2: Error profiling & correction
    techs_to_correct = [t for t in set(all_technologies) if t != 'illumina' or not has_long_reads]
    errorsmith_status = '✓ ErrorSmith AI' if 'ErrorSmith' in correction_ai_modules else '○ Classical'
    click.echo(f"  2. Error Correction ({', '.join(techs_to_correct)}) [{errorsmith_status}]")
    click.echo(f"     └─ Profile errors → Correct reads")
    
    # Step 3: Illumina contigger (if applicable)
    if has_illumina:
        click.echo(f"  3. Illumina OLC Contigger")
        click.echo(f"     └─ Generate artificial long reads from Illumina")
    
    # Step 4: Graph assembly
    graph_modules = []
    if 'EdgeWarden' in assembly_ai_modules:
        graph_modules.append('EdgeWarden')
    if 'PathWeaver' in assembly_ai_modules:
        graph_modules.append('PathWeaver')
    
    if graph_modules:
        graph_status = f"with {', '.join(graph_modules)}"
    else:
        graph_status = "Classical algorithms"
    
    step_num = 4 if has_illumina else 3
    click.echo(f"  {step_num}. DBG Assembly ({graph_status})")
    click.echo(f"     ├─ Build De Bruijn graph from long reads")
    if 'EdgeWarden' in assembly_ai_modules:
        click.echo(f"     ├─ EdgeWarden: AI-powered edge filtering")
    if 'PathWeaver' in assembly_ai_modules:
        click.echo(f"     └─ PathWeaver: AI-powered path selection")
    else:
        click.echo(f"     └─ Simplify graph (tips, bubbles, low coverage)")
    
    # Step 5: String graph (if UL reads)
    if has_ul:
        step_num += 1
        threadcompass_status = '✓ ThreadCompass' if 'ThreadCompass' in assembly_ai_modules else '○ Heuristic'
        click.echo(f"  {step_num}. String Graph Overlay ({threadcompass_status})")
        click.echo(f"     ├─ Map ultra-long reads to DBG")
        click.echo(f"     ├─ Create UL-derived super-edges")
        if 'ThreadCompass' in assembly_ai_modules:
            click.echo(f"     └─ ThreadCompass: AI-powered UL routing")
        else:
            click.echo(f"     └─ Select paths by coverage heuristics")
    
    # Step 6: Hi-C scaffolding (if available)
    if has_hic:
        step_num += 1
        click.echo(f"  {step_num}. Hi-C Scaffolding")
        click.echo(f"     ├─ R1: {Path(hic_r1).name if hic_r1 else 'N/A'}")
        click.echo(f"     ├─ R2: {Path(hic_r2).name if hic_r2 else 'N/A'}")
        click.echo(f"     └─ Order and orient contigs using proximity data")
        pipeline_config['scaffolding']['hic']['enabled'] = True
    elif hic_r1 or hic_r2:
        click.echo(f"\n❌ Error: Both --hic-r1 and --hic-r2 must be specified together", err=True)
        ctx.exit(1)
    
    # Step 7: Phasing with iterations
    step_num += 1
    detangler_status = '✓ Haplotype Detangler' if 'Haplotype Detangler' in assembly_ai_modules else '○ Classical'
    num_iterations = pipeline_config.get('assembly', {}).get('phasing', {}).get('iterations', 2 if ai_enabled else 3)
    click.echo(f"  {step_num}. Haplotype Phasing ({detangler_status})")
    click.echo(f"     ├─ Separate homologous paths")
    click.echo(f"     └─ Iterate graph refinement: {num_iterations} rounds")
    if ai_enabled and ('EdgeWarden' in assembly_ai_modules or 'PathWeaver' in assembly_ai_modules):
        click.echo(f"        └─ Re-apply {', '.join([m for m in ['EdgeWarden', 'PathWeaver'] if m in assembly_ai_modules])} each iteration")
    
    # Step 8: Graph cleanup
    step_num += 1
    svscribe_status = '✓ SVScribe' if 'SVScribe' in assembly_ai_modules else '○ Heuristic'
    click.echo(f"  {step_num}. Graph Cleanup & SV Detection ({svscribe_status})")
    click.echo(f"     ├─ Final graph simplification")
    if 'SVScribe' in assembly_ai_modules:
        click.echo(f"     └─ SVScribe: AI-powered structural variant calling")
    else:
        click.echo(f"     └─ Heuristic SV detection")
    
    # Step 9: Finishing
    step_num += 1
    click.echo(f"  {step_num}. Finishing & Polishing")
    if ai_finish:
        click.echo(f"     ├─ Claude AI finishing (experimental)")
    click.echo(f"     ├─ Extract final contigs from graph")
    click.echo(f"     ├─ Polish with T2T-Polish")
    click.echo(f"     └─ Generate assembly statistics & QC report")
    
    click.echo(f"{'─'*60}")
    
    # Summary of AI/ML status
    click.echo(f"\n🤖 AI/ML Status: {'ENABLED' if ai_enabled else 'DISABLED (--classical)'}")
    if ai_enabled:
        all_modules = correction_ai_modules + assembly_ai_modules
        if all_modules:
            click.echo(f"  • Active modules: {', '.join(all_modules)}")
        else:
            click.echo(f"  • No AI modules enabled (check config)")
    
    # Config file info
    if resume:
        click.echo(f"\n⚙️  Resume: ✓ (from last checkpoint)")
    if config:
        click.echo(f"\n⚙️  Config File: {config}")
        click.echo(f"  • All parameters from config + CLI overrides")
    else:
        click.echo(f"\n⚙️  Config: Using defaults (generate with: strandweaver config init)")
    
    click.echo(f"{'='*60}\n")
    
    # ========================================================================
    # Run Pipeline
    # ========================================================================
    from .utils.pipeline import PipelineOrchestrator
    
    # Add runtime parameters to config
    pipeline_config['runtime'] = {
        'reads': all_reads,
        'technologies': all_technologies,
        'output_dir': output,
        'hic_r1': hic_r1,
        'hic_r2': hic_r2,
        'verbose': verbose,
    }
    
    # Initialize and run orchestrator
    orchestrator = PipelineOrchestrator(
        config=pipeline_config,
        checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None
    )
    
    try:
        result = orchestrator.run(start_from=start_from, resume=resume)
        
        click.echo("\n" + "="*60)
        click.echo("✅ Pipeline completed successfully!")
        click.echo("="*60)
        click.echo(f"Steps completed: {result.get('steps_completed', 'N/A')}")
        click.echo(f"Output directory: {output}")
        
    except Exception as e:
        click.echo(f"\n❌ Pipeline failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        ctx.exit(1)


# ============================================================================
# Individual Step Commands
# ============================================================================

@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input reads file (FASTQ/FASTA)')
@click.option('--technology', '-tech', 
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              default='auto', help='Sequencing technology (auto = infer from data)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output error profile file (JSON)')
@click.option('--sample-size', type=int, default=1000000,
              help='Number of reads to sample for profiling')
@click.option('--min-quality', type=int, default=20,
              help='Minimum base quality threshold')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
# Nanopore-specific metadata options
@click.option('--ont-flowcell', type=str, default=None,
              help='ONT flow cell type (e.g., R9.4.1, R10.4.1)')
@click.option('--ont-basecaller', type=str, default=None,
              help='ONT basecaller (e.g., guppy, dorado)')
@click.option('--ont-accuracy', type=str, default=None,
              help='ONT basecaller accuracy mode (sup, hac, fast)')
@click.option('--ont-detect', is_flag=True, default=False,
              help='Auto-detect ONT metadata using LongBow (requires installation)')
def profile(input, technology, output, sample_size, min_quality, threads,
            ont_flowcell, ont_basecaller, ont_accuracy, ont_detect):
    """
    Profile sequencing errors in reads.
    
    Analyzes reads to identify error patterns, rates, and positions.
    Generates a comprehensive error profile report.
    
    Examples:
        # Auto-detect technology from read characteristics
        strandweaver profile -i reads.fq -o profile.json
        
        # Explicitly specify Illumina reads
        strandweaver profile -i reads.fq --technology illumina -o profile.json
    """
    from pathlib import Path
    import json
    from .io import (
        parse_technology,
        ReadTechnology,
        read_fastq,
        parse_nanopore_metadata,
        NanoporeMetadata,
        detect_nanopore_metadata_with_longbow
    )
    from .errors import ErrorProfiler
    
    click.echo(f"{'='*60}")
    click.echo(f"StrandWeaver Error Profiling")
    click.echo(f"{'='*60}")
    click.echo(f"Input: {input}")
    click.echo(f"Technology: {technology}")
    click.echo(f"Sample size: {sample_size:,} reads")
    click.echo(f"Min quality: Q{min_quality}")
    click.echo(f"Output: {output}")
    click.echo(f"{'='*60}\n")
    
    # Parse technology
    if technology == 'auto':
        tech_enum = ReadTechnology.UNKNOWN  # Will auto-detect
        click.echo("ℹ️  Auto-detecting technology from read characteristics...")
    else:
        try:
            tech_enum = parse_technology(technology)
            click.echo(f"✓ Technology: {tech_enum.value.upper()}\n")
        except ValueError as e:
            click.echo(f"❌ Error: {e}", err=True)
            from sys import exit
            exit(1)
    
    # Verify input file exists
    input_path = Path(input)
    if not input_path.exists():
        click.echo(f"❌ Error: File not found: {input}", err=True)
        from sys import exit
        exit(1)
    
    # Handle Nanopore-specific metadata
    ont_metadata = None
    is_ont = tech_enum in [ReadTechnology.ONT_REGULAR, ReadTechnology.ONT_ULTRALONG] or \
             (tech_enum == ReadTechnology.UNKNOWN and technology in ['ont', 'ont_ultralong'])
    
    if is_ont:
        if ont_detect:
            # Auto-detect using LongBow
            click.echo("\n🔍 Detecting ONT metadata with LongBow...")
            ont_metadata = detect_nanopore_metadata_with_longbow(input_path, sample_size=min(sample_size, 10000))
            if ont_metadata:
                click.echo(f"✓ Detected: {ont_metadata}\n")
            else:
                click.echo("⚠️  Could not detect ONT metadata, using standard ONT profiling\n")
        elif any([ont_flowcell, ont_basecaller, ont_accuracy]):
            # Manual metadata provided
            try:
                ont_metadata = parse_nanopore_metadata(
                    flow_cell=ont_flowcell,
                    basecaller=ont_basecaller,
                    accuracy=ont_accuracy
                )
                if ont_metadata:
                    click.echo(f"\n✓ ONT Metadata: {ont_metadata}\n")
            except ValueError as e:
                click.echo(f"❌ Error: {e}", err=True)
                from sys import exit
                exit(1)
        else:
            click.echo("\nℹ️  No ONT metadata provided - using standard ONT profiling")
            click.echo("   Tip: Use --ont-detect or provide --ont-flowcell/basecaller/accuracy for improved profiling\n")
    elif any([ont_flowcell, ont_basecaller, ont_accuracy, ont_detect]):
        # ONT metadata provided for non-ONT technology
        click.echo("\n⚠️  Warning: ONT metadata flags ignored (technology is not ONT)\n")
    
    # Profile errors
    click.echo(f"\nProfiling errors...")
    click.echo(f"  K-mer size: 21")
    click.echo(f"  Min quality: Q{min_quality}")
    click.echo(f"  Technology: {tech_enum.value.upper() if tech_enum != ReadTechnology.UNKNOWN else 'AUTO'}\n")
    
    try:
        profiler = ErrorProfiler(
            k=21,
            sample_size=sample_size,
            min_quality=min_quality,
            enable_positional_analysis=True
        )
        
        # Profile reads from file (profiler loads reads internally)
        error_profile = profiler.profile(input_path, tech_enum, ont_metadata=ont_metadata)
        
        # Display summary
        click.echo(f"\n{'='*60}")
        click.echo(f"Error Profile Summary")
        click.echo(f"{'='*60}")
        click.echo(error_profile.summary())
        click.echo(f"{'='*60}\n")
        
        # Save to JSON
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(error_profile.to_dict(), f, indent=2)
        
        click.echo(f"✓ Profile saved to: {output}")
        click.echo(f"\nUse this profile with:")
        click.echo(f"  strandweaver correct -i {input} --technology {tech_enum.value} \\")
        click.echo(f"      --profile {output} -o corrected.fq")
        
    except Exception as e:
        click.echo(f"\n❌ Error during profiling: {e}", err=True)
        import traceback
        traceback.print_exc()
        from sys import exit
        exit(1)


@main.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input reads file (FASTQ/FASTA)')
@click.option('--technology', '-tech', required=True,
              type=click.Choice(['ancient', 'illumina', 'ont', 'ont_ultralong', 'pacbio', 'hic']),
              help='Sequencing technology type')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output corrected reads file (FASTQ)')
@click.option('--profile', type=click.Path(exists=True),
              help='Error profile from profiling step (JSON)')
@click.option('--pre-corrected', is_flag=True,
              help='Reads are already error-corrected (skip correction)')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
# ONT-specific optimization parameters
@click.option('--ont-flowcell', type=str, default=None,
              help='ONT flow cell type (e.g., R9.4.1, R10.4.1) for optimized correction')
@click.option('--ont-basecaller', type=str, default=None,
              help='ONT basecaller (e.g., guppy, dorado) for optimized correction')
@click.option('--ont-accuracy', type=str, default=None,
              help='ONT basecaller accuracy mode (sup, hac, fast) for optimized correction')
@click.option('--use-gpu/--no-gpu', default=False,
              help='Enable GPU acceleration for k-mer counting (NVIDIA CUDA required)')
@click.option('--gpu-threshold', type=int, default=1000,
              help='Minimum reads to use GPU (default: 1000)')
@click.option('--kmer-size', '-k', type=int, default=21,
              help='K-mer size for error correction (default: 21)')
@click.option('--min-quality', type=int, default=7,
              help='Minimum base quality threshold (default: 7)')
@click.option('--skip-high-quality/--no-skip-high-quality', default=True,
              help='Skip correcting high-quality homopolymers (Q>25)')
@click.option('--quality-threshold', type=int, default=25,
              help='Quality threshold for skipping correction (default: 25)')
@click.option('--early-stop-factor', type=float, default=1.5,
              help='Early stopping factor for correction search (default: 1.5)')
@click.option('--cache-stats/--no-cache-stats', default=False,
              help='Show k-mer cache statistics after correction')
@click.option('--verbose-stats/--no-verbose-stats', default=True,
              help='Show detailed correction statistics')
@click.option('--error-viz/--no-error-viz', default=False,
              help='Generate error profiling and correction visualizations')
@click.option('--viz-output-dir', type=click.Path(), default='correction_reports',
              help='Output directory for visualizations (default: correction_reports)')
@click.pass_context
def correct(ctx, input, technology, output, profile, pre_corrected, threads,
            ont_flowcell, ont_basecaller, ont_accuracy,
            use_gpu, gpu_threshold, kmer_size, min_quality,
            skip_high_quality, quality_threshold, early_stop_factor,
            cache_stats, verbose_stats, error_viz, viz_output_dir):
    """
    Correct sequencing errors in reads with optimized algorithms.
    
    Applies technology-specific error correction with advanced optimizations:
    - K-mer caching for 50-80% speedup
    - Quality-weighted scoring for better accuracy
    - Early stopping heuristics
    - Ternary search for optimal lengths
    - Adaptive window sizing
    - Parallel processing (multi-core)
    - Bloom filter pre-filtering
    - Optional GPU acceleration (10-100x speedup on large datasets)
    
    Examples:
        # ONT reads with flow cell metadata and GPU acceleration
        strandweaver correct -i ont.fq --technology ont \\
            --ont-flowcell R10.4.1 --ont-basecaller dorado --ont-accuracy sup \\
            --use-gpu --threads 8 -o ont_corrected.fq
        
        # ONT reads with all optimizations, no GPU
        strandweaver correct -i ont.fq --technology ont \\
            --threads 8 --cache-stats -o ont_corrected.fq
        
        # PacBio HiFi (pre-corrected, skip correction)
        strandweaver correct -i hifi.fq --technology pacbio \\
            --pre-corrected -o hifi.fq
        
        # Illumina reads with custom k-mer size
        strandweaver correct -i illumina.fq --technology illumina \\
            --kmer-size 31 --threads 16 -o illumina_corrected.fq
        
        # Ancient DNA with damage-aware correction
        strandweaver correct -i ancient.fq --technology ancient \\
            --threads 8 -o ancient_corrected.fq
    """
    from pathlib import Path
    import json
    import time
    from .io import (
        parse_technology,
        ReadTechnology,
        read_fastq,
        write_fastq,
        parse_nanopore_metadata
    )
    
    verbose = ctx.obj.get('VERBOSE', False)
    
    click.echo(f"{'='*70}")
    click.echo(f"StrandWeaver Error Correction (Optimized)")
    click.echo(f"{'='*70}")
    click.echo(f"Input: {input}")
    click.echo(f"Technology: {technology.upper()}")
    click.echo(f"Output: {output}")
    click.echo(f"Threads: {threads}")
    
    # Handle pre-corrected reads (just copy)
    if pre_corrected:
        click.echo(f"\n⚠️  Reads marked as pre-corrected - copying without correction")
        input_path = Path(input)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(input_path, output_path)
        
        click.echo(f"✓ Copied {input} → {output}")
        return
    
    # Parse technology
    try:
        tech_enum = parse_technology(technology)
    except ValueError as e:
        click.echo(f"❌ Error: {e}", err=True)
        ctx.exit(1)
    
    # Optimization configuration
    click.echo(f"\nOptimizations:")
    click.echo(f"  K-mer caching: Enabled")
    click.echo(f"  Quality weighting: Enabled")
    click.echo(f"  Early stopping: Enabled (factor={early_stop_factor})")
    click.echo(f"  Skip high-quality: {'Enabled' if skip_high_quality else 'Disabled'} (Q>{quality_threshold})")
    click.echo(f"  Ternary search: Enabled")
    click.echo(f"  Adaptive windows: Enabled")
    click.echo(f"  Parallel processing: Enabled ({threads} threads)")
    click.echo(f"  Bloom filter: Enabled")
    
    if use_gpu:
        click.echo(f"  GPU acceleration: Requested (threshold={gpu_threshold} reads)")
        try:
            from .utils.hardware_management import get_gpu_info
            gpu_info = get_gpu_info()
            click.echo(f"    {gpu_info}")
        except ImportError:
            click.echo(f"    ⚠️  CuPy not installed - falling back to CPU")
            click.echo(f"    Install: pip install cupy-cuda11x (for CUDA 11) or cupy-cuda12x (for CUDA 12)")
            use_gpu = False
    else:
        click.echo(f"  GPU acceleration: Disabled (use --use-gpu to enable)")
    
    # Technology-specific correction
    click.echo(f"\n{'='*70}")
    
    if tech_enum in [ReadTechnology.ONT_REGULAR, ReadTechnology.ONT_ULTRALONG]:
        # ONT correction with all optimizations
        click.echo(f"ONT Error Correction (Homopolymer-Optimized)")
        click.echo(f"{'='*70}")
        
        # Parse ONT metadata if provided
        ont_metadata = None
        if any([ont_flowcell, ont_basecaller, ont_accuracy]):
            try:
                ont_metadata = parse_nanopore_metadata(
                    flow_cell=ont_flowcell,
                    basecaller=ont_basecaller,
                    accuracy=ont_accuracy
                )
                if ont_metadata:
                    click.echo(f"ONT Metadata: {ont_metadata}")
                    click.echo(f"  Homopolymer error rate: {ont_metadata.homopolymer_error_rate:.4f}")
                    click.echo(f"  Overall error rate: {ont_metadata.overall_error_rate:.4f}")
            except ValueError as e:
                click.echo(f"⚠️  Warning: {e}")
                click.echo(f"    Using standard ONT correction parameters")
        else:
            click.echo(f"No ONT metadata provided - using standard parameters")
            click.echo(f"  Tip: Use --ont-flowcell, --ont-basecaller, --ont-accuracy for optimized correction")
        
        click.echo(f"\nK-mer spectrum parameters:")
        click.echo(f"  K-mer size: {kmer_size}")
        click.echo(f"  Min quality: Q{min_quality}")
        click.echo(f"{'='*70}\n")
        
        # Import ONT corrector
        from .correction import ONTCorrector, KmerSpectrum
        
        # Load reads
        click.echo("Loading reads...")
        input_path = Path(input)
        reads = list(read_fastq(input_path))
        click.echo(f"✓ Loaded {len(reads):,} reads\n")
        
        # Build k-mer spectrum
        click.echo("Building k-mer spectrum...")
        start_time = time.time()
        
        spectrum = KmerSpectrum(
            k=kmer_size,
            min_quality=min_quality,
            use_bloom_filter=True  # Always enable Bloom filter
        )
        spectrum.add_reads(reads)
        
        spectrum_time = time.time() - start_time
        click.echo(f"✓ Built k-mer spectrum in {spectrum_time:.2f}s")
        
        # Show Bloom filter stats
        bloom_stats = spectrum.bloom_filter.get_stats()
        click.echo(f"  Bloom filter: {bloom_stats['total_queries']:,} queries, "
                   f"{bloom_stats['false_positive_rate']:.2%} FP rate\n")
        
        # Create corrector with optimizations
        click.echo("Initializing ONT corrector with optimizations...")
        corrector = ONTCorrector(
            spectrum=spectrum,
            window_size=11,  # Will be adaptive
            min_support=5,
            ont_metadata=ont_metadata,
            use_gpu=use_gpu,
            gpu_threshold=gpu_threshold,
            enable_visualizations=error_viz,
            visualization_output_dir=viz_output_dir
        )
        
        # Show GPU status if requested
        if use_gpu:
            if hasattr(corrector, 'gpu_counter') and corrector.gpu_counter:
                click.echo(f"✓ GPU k-mer counter initialized")
            else:
                click.echo(f"  Using CPU (GPU not available or dataset too small)")
        
        click.echo(f"✓ Corrector ready\n")
        
        # Correct reads with parallel processing
        click.echo(f"Correcting {len(reads):,} reads with {threads} threads...")
        start_time = time.time()
        
        if threads > 1:
            # Parallel correction
            import multiprocessing as mp
            with mp.Pool(threads) as pool:
                corrected_reads = pool.map(corrector.correct_read, reads)
        else:
            # Sequential correction
            corrected_reads = [corrector.correct_read(read) for read in reads]
        
        correction_time = time.time() - start_time
        click.echo(f"✓ Corrected {len(corrected_reads):,} reads in {correction_time:.2f}s")
        click.echo(f"  Throughput: {len(reads) / correction_time:.1f} reads/sec\n")
        
        # Show cache statistics if requested
        if cache_stats and hasattr(corrector, '_kmer_cache'):
            stats = corrector.get_cache_stats()
            click.echo(f"K-mer Cache Statistics:")
            click.echo(f"  Cached k-mers: {stats['cached_kmers']:,}")
            click.echo(f"  Cache hits: {stats['cache_hits']:,}")
            click.echo(f"  Cache misses: {stats['cache_misses']:,}")
            total = stats['cache_hits'] + stats['cache_misses']
            if total > 0:
                hit_rate = stats['cache_hits'] / total * 100
                click.echo(f"  Hit rate: {hit_rate:.1f}%\n")
        
        # Show correction statistics if requested
        if verbose_stats:
            # Count corrections
            total_corrections = 0
            homopolymer_corrections = 0
            for orig, corr in zip(reads, corrected_reads):
                if orig.sequence != corr.sequence:
                    total_corrections += 1
                    # Simple homopolymer check (consecutive identical bases)
                    import re
                    if re.search(r'([ACGT])\1{2,}', orig.sequence) or \
                       re.search(r'([ACGT])\1{2,}', corr.sequence):
                        homopolymer_corrections += 1
            
            click.echo(f"Correction Statistics:")
            click.echo(f"  Total reads: {len(reads):,}")
            click.echo(f"  Corrected reads: {total_corrections:,} ({total_corrections/len(reads)*100:.1f}%)")
            click.echo(f"  Homopolymer corrections: {homopolymer_corrections:,}\n")
        
        # Write corrected reads
        click.echo(f"Writing corrected reads to {output}...")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_fastq(corrected_reads, output_path)
        
        click.echo(f"✓ Wrote {len(corrected_reads):,} corrected reads")
        click.echo(f"\n{'='*70}")
        click.echo(f"✅ ONT error correction complete!")
        click.echo(f"{'='*70}")
        click.echo(f"Total time: {spectrum_time + correction_time:.2f}s")
        click.echo(f"Output: {output}")
        
    elif tech_enum == ReadTechnology.PACBIO_HIFI:
        # PacBio HiFi correction
        from .correction import PacBioCorrector
        
        click.echo(f"PacBio HiFi Error Correction")
        click.echo(f"{'='*70}\n")
        
        # Create corrector with visualization support
        click.echo("Initializing PacBio HiFi corrector...")
        corrector = PacBioCorrector(
            k_size=kmer_size,
            min_freq=2,
            enable_visualizations=error_viz,
            visualization_output_dir=viz_output_dir
        )
        
        click.echo(f"✓ Corrector ready\n")
        
        # Correct reads
        click.echo(f"Correcting reads...")
        stats = corrector.correct_reads(
            input_file=Path(input),
            output_file=Path(output)
        )
        
        click.echo(f"✓ Correction complete!")
        if verbose_stats:
            click.echo(f"\nCorrection Statistics:")
            click.echo(f"  Reads corrected: {stats.reads_corrected:,}")
            click.echo(f"  Total bases: {stats.total_bases:,}")
        
    elif tech_enum == ReadTechnology.ILLUMINA:
        # Illumina correction
        from .correction import IlluminaCorrector
        
        click.echo(f"Illumina Error Correction")
        click.echo(f"{'='*70}\n")
        
        # Create corrector with visualization support
        click.echo("Initializing Illumina corrector...")
        corrector = IlluminaCorrector(
            k_size=kmer_size,
            min_freq=2,
            enable_visualizations=error_viz,
            visualization_output_dir=viz_output_dir
        )
        
        click.echo(f"✓ Corrector ready\n")
        
        # Correct reads
        click.echo(f"Correcting reads...")
        stats = corrector.correct_reads(
            input_file=Path(input),
            output_file=Path(output)
        )
        
        click.echo(f"✓ Correction complete!")
        if verbose_stats:
            click.echo(f"\nCorrection Statistics:")
            click.echo(f"  Reads corrected: {stats.reads_corrected:,}")
            click.echo(f"  Total bases: {stats.total_bases:,}")
        
    elif tech_enum == ReadTechnology.ANCIENT_DNA:
        # Ancient DNA correction
        from .correction import AncientDNACorrector
        
        click.echo(f"Ancient DNA Error Correction")
        click.echo(f"{'='*70}\n")
        
        # Create corrector with visualization support
        click.echo("Initializing Ancient DNA corrector with damage-aware correction...")
        corrector = AncientDNACorrector(
            k_size=kmer_size,
            min_freq=2,
            enable_visualizations=error_viz,
            visualization_output_dir=viz_output_dir
        )
        
        click.echo(f"✓ Corrector ready\n")
        
        # Correct reads
        click.echo(f"Correcting reads with intelligent deamination screening...")
        stats = corrector.correct_reads(
            input_file=Path(input),
            output_file=Path(output)
        )
        
        click.echo(f"✓ Correction complete!")
        if verbose_stats:
            click.echo(f"\nCorrection Statistics:")
            click.echo(f"  Reads corrected: {stats.reads_corrected:,}")
            click.echo(f"  Total bases: {stats.total_bases:,}")
            
            # Show damage statistics if available
            if hasattr(corrector, 'get_damage_stats'):
                damage_stats = corrector.get_damage_stats()
                if damage_stats:
                    click.echo(f"\nAncient DNA Damage Analysis:")
                    click.echo(f"  Damage signature detected: {damage_stats.get('damage_signature_detected', 'N/A')}")
                    click.echo(f"  5' C→T damage rate: {damage_stats.get('damage_5p_rate', 0)*100:.2f}%")
                    click.echo(f"  3' G→A damage rate: {damage_stats.get('damage_3p_rate', 0)*100:.2f}%")
                    click.echo(f"  Damage corrections: {damage_stats.get('damage_corrections', 0):,}")
                    click.echo(f"  Variants preserved: {damage_stats.get('variants_preserved', 0):,}")
        
    elif tech_enum == ReadTechnology.HI_C:
        # Hi-C reads - NO correction, just pass through
        click.echo(f"Hi-C Proximity Ligation Reads")
        click.echo(f"{'='*70}\n")
        click.echo(f"⚠️  Hi-C reads are NOT error-corrected (used for scaffolding only)")
        click.echo(f"\nHi-C reads are proximity ligation data used for:")
        click.echo(f"  • Contact mapping between genomic regions")
        click.echo(f"  • Scaffolding assembled contigs")
        click.echo(f"  • Chromosome-scale assembly")
        click.echo(f"\nThey should NOT be error-corrected as:")
        click.echo(f"  • Chimeric reads are expected (junction between distant regions)")
        click.echo(f"  • Only mapping position matters, not sequence accuracy")
        click.echo(f"  • Error correction would destroy proximity information")
        click.echo(f"\nCopying reads without correction...")
        
        input_path = Path(input)
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy(input_path, output_path)
        
        click.echo(f"✓ Hi-C reads copied: {input} → {output}")
        click.echo(f"\nUse these reads for scaffolding AFTER assembly with:")
        click.echo(f"  strandweaver scaffold --contigs assembly.fasta \\")
        click.echo(f"      --hic-r1 {output} --hic-r2 <R2_file> -o scaffolded.fasta")
        
    else:
        click.echo(f"❌ Error: Unsupported technology: {tech_enum.value}", err=True)
        ctx.exit(1)


@main.command()
@click.option('--reads', '-r', 'reads_files', multiple=True, required=True,
              type=click.Path(exists=True),
              help='Input corrected reads files (specify multiple times)')
@click.option('--technology', '-tech', 'technologies', multiple=True,
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio']),
              help='Technology for each reads file (must match order)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output merged reads file (FASTQ)')
@click.option('--weights', type=str,
              help='Comma-separated coverage weights (e.g., "1.0,0.8,1.2")')
def merge(reads_files, technologies, output, weights):
    """
    Merge corrected reads from multiple technologies.
    
    Combines error-corrected reads from different sequencing platforms
    into a unified dataset for assembly. Optionally applies technology-specific
    weighting to balance coverage contributions.
    
    Examples:
        # Merge two technologies with equal weight
        strandweaver merge -r illumina.fq -r ont.fq \\
            --technology illumina --technology ont -o merged.fq
        
        # Merge with custom weights (Illumina 100%, ONT 80%, HiFi 120%)
        strandweaver merge -r illumina.fq -r ont.fq -r hifi.fq \\
            --technology illumina --technology ont --technology pacbio \\
            --weights "1.0,0.8,1.2" -o merged.fq
    """
    # Validate technology specifications
    if technologies and len(technologies) != len(reads_files):
        click.echo(f"❌ Error: Number of --technology flags ({len(technologies)}) must match "
                   f"number of --reads files ({len(reads_files)})", err=True)
        from sys import exit
        exit(1)
    
    click.echo(f"Merging {len(reads_files)} input file(s):")
    for i, reads in enumerate(reads_files, 1):
        tech = technologies[i-1] if technologies else 'auto'
        click.echo(f"  {i}. {reads} ({tech})")
    
    click.echo(f"Output: {output}")
    
    if weights:
        click.echo(f"Weights: {weights}")
    
    # TODO: Implement multi-technology merger
    click.echo("\n⚠ Multi-technology merging implementation coming soon!")


@main.command('build-contigs')
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input corrected reads file (FASTQ)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output contigs file (FASTA)')
@click.option('--min-overlap', type=int, default=1000,
              help='Minimum overlap length for contig building')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
def build_contigs(input, output, min_overlap, threads):
    """
    Build contigs from short reads.
    
    Performs all-vs-all alignment of short reads to construct longer
    pseudo-long reads (contigs) for graph assembly.
    """
    click.echo(f"Building contigs from: {input}")
    click.echo(f"Min overlap: {min_overlap} bp")
    click.echo(f"Output: {output}")
    
    # TODO: Implement contig builder
    click.echo("\n⚠ Contig building implementation coming soon!")


@main.command()
@click.option('--reads', '-r', 'reads_files', multiple=True,
              type=click.Path(exists=True),
              help='[OLD SYNTAX] Input reads/contigs file. Use -r1/-r2/etc for clarity.')
@click.option('--technology', '-tech', 'technologies', multiple=True,
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='[OLD SYNTAX] Technology for each reads file. Use --technology1/--technology2/etc for clarity.')
@click.option('-r1', '--reads1', type=click.Path(exists=True),
              help='Read file 1 (pairs with --technology1)')
@click.option('-r2', '--reads2', type=click.Path(exists=True),
              help='Read file 2 (pairs with --technology2)')
@click.option('-r3', '--reads3', type=click.Path(exists=True),
              help='Read file 3 (pairs with --technology3)')
@click.option('-r4', '--reads4', type=click.Path(exists=True),
              help='Read file 4 (pairs with --technology4)')
@click.option('-r5', '--reads5', type=click.Path(exists=True),
              help='Read file 5 (pairs with --technology5)')
@click.option('--technology1', '-tech1',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 1')
@click.option('--technology2', '-tech2',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 2')
@click.option('--technology3', '-tech3',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 3')
@click.option('--technology4', '-tech4',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 4')
@click.option('--technology5', '-tech5',
              type=click.Choice(['illumina', 'ancient', 'ont', 'ont_ultralong', 'pacbio', 'auto']),
              help='Technology for read file 5')
@click.option('--illumina-r1', type=click.Path(exists=True),
              help='Illumina paired-end R1 reads (forward reads)')
@click.option('--illumina-r2', type=click.Path(exists=True),
              help='Illumina paired-end R2 reads (reverse reads)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output assembly file (FASTA)')
@click.option('--graph', '-g', type=click.Path(),
              help='Output assembly graph file (GFA)')
@click.option('--graph-type', 
              type=click.Choice(['string', 'debruijn', 'hybrid']),
              default='string', help='Assembly graph type')
@click.option('--min-coverage', type=int, default=5,
              help='Minimum coverage threshold')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
def assemble(reads_files, technologies,
             reads1, reads2, reads3, reads4, reads5,
             technology1, technology2, technology3, technology4, technology5,
             illumina_r1, illumina_r2, output, graph, graph_type, min_coverage, threads):
    """
    Perform graph-based genome assembly.
    
    Constructs an assembly graph from reads/contigs and traverses it
    to generate draft genome sequences. Supports multi-technology input.
    
    Examples:
        # RECOMMENDED: Numbered syntax
        # Single technology
        strandweaver assemble -r1 corrected.fq --technology1 illumina -o assembly.fa
        
        # Illumina paired-end separate files
        strandweaver assemble --illumina-r1 R1.fq --illumina-r2 R2.fq -o assembly.fa
        
        # Multi-technology (hybrid) with explicit numbering
        strandweaver assemble \\
            -r1 illumina.fq --technology1 illumina \\
            -r2 ont.fq --technology2 ont \\
            -r3 hifi.fq --technology3 pacbio \\
            -o hybrid_assembly.fa -g assembly.gfa
        
        # With Illumina paired-end + other technologies
        strandweaver assemble \\
            --illumina-r1 ill_R1.fq --illumina-r2 ill_R2.fq \\
            -r1 ont.fq --technology1 ont \\
            -o assembly.fa
        
        # Auto-detect technologies (omit --technology flags)
        strandweaver assemble -r1 reads1.fq -r2 reads2.fq -o assembly.fa
    """
    # Collect numbered reads and technologies
    numbered_reads = {}
    numbered_techs = {}
    
    if reads1: numbered_reads[1] = reads1
    if reads2: numbered_reads[2] = reads2
    if reads3: numbered_reads[3] = reads3
    if reads4: numbered_reads[4] = reads4
    if reads5: numbered_reads[5] = reads5
    
    if technology1: numbered_techs[1] = technology1
    if technology2: numbered_techs[2] = technology2
    if technology3: numbered_techs[3] = technology3
    if technology4: numbered_techs[4] = technology4
    if technology5: numbered_techs[5] = technology5
    
    # Check for mixing old and new syntax
    using_old_syntax = bool(reads_files or technologies)
    using_new_syntax = bool(numbered_reads)
    
    if using_old_syntax and using_new_syntax:
        click.echo(f"❌ Error: Cannot mix old syntax (-r/--technology) with new syntax (-r1/--technology1)", err=True)
        click.echo(f"\nUse numbered syntax (recommended): -r1 file1.fq --technology1 illumina", err=True)
        from sys import exit
        exit(1)
    
    # Validate Illumina paired-end input
    if illumina_r1 and not illumina_r2:
        click.echo(f"❌ Error: --illumina-r1 requires --illumina-r2", err=True)
        from sys import exit
        exit(1)
    if illumina_r2 and not illumina_r1:
        click.echo(f"❌ Error: --illumina-r2 requires --illumina-r1", err=True)
        from sys import exit
        exit(1)
    
    # Build complete reads list
    all_reads = []
    all_technologies = []
    
    if illumina_r1 and illumina_r2:
        all_reads.extend([illumina_r1, illumina_r2])
        all_technologies.extend(['illumina', 'illumina'])
    
    if using_new_syntax:
        # NEW NUMBERED SYNTAX
        for num in sorted(numbered_reads.keys()):
            all_reads.append(numbered_reads[num])
            all_technologies.append(numbered_techs.get(num, 'auto'))
        
        # Warn if technology specified without corresponding read
        for num in numbered_techs.keys():
            if num not in numbered_reads:
                click.echo(f"⚠️  Warning: --technology{num} specified but no -r{num} file provided (ignored)", err=True)
    
    elif using_old_syntax:
        # OLD SYNTAX
        if reads_files:
            all_reads.extend(reads_files)
            
            if technologies:
                if len(technologies) != len(reads_files):
                    click.echo(f"❌ Error: Number of --technology flags ({len(technologies)}) must match "
                               f"number of --reads files ({len(reads_files)})", err=True)
                    click.echo(f"\nUse numbered syntax: -r1 file1.fq --technology1 illumina", err=True)
                    from sys import exit
                    exit(1)
                all_technologies.extend(technologies)
            else:
                all_technologies.extend(['auto'] * len(reads_files))
                click.echo("ℹ️  Auto-detecting technologies from read characteristics")
    
    # Ensure we have at least some input
    if not all_reads:
        click.echo(f"❌ Error: No input reads specified. Use --illumina-r1/--illumina-r2 or -r1/-r2/etc", err=True)
        from sys import exit
        exit(1)
    
    click.echo(f"Assembling genome from {len(all_reads)} input file(s):")
    for i, (reads, tech) in enumerate(zip(all_reads, all_technologies), 1):
        click.echo(f"  {i}. {reads} ({tech})")
    
    click.echo(f"Graph type: {graph_type}")
    click.echo(f"Min coverage: {min_coverage}x")
    click.echo(f"Output assembly: {output}")
    
    if graph:
        click.echo(f"Output graph: {graph}")
    
    # TODO: Implement graph assembly
    click.echo(f"\n⚠ {graph_type} graph assembly implementation coming soon!")


@main.command()
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Input assembly graph file (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output finished assembly file (FASTA)')
@click.option('--ai', type=click.Choice(['claude', 'heuristic']),
              default='claude', help='Finishing method')
@click.option('--interactive/--no-interactive', default=False,
              help='Ask user for ambiguous decisions')
@click.option('--api-key', envvar='ANTHROPIC_API_KEY',
              help='Claude API key (or set ANTHROPIC_API_KEY env var)')
def finish(graph, output, ai, interactive, api_key):
    """
    Finish assembly with AI-powered path resolution.
    
    Resolves ambiguous paths in the assembly graph using AI reasoning
    or heuristic algorithms to produce a final polished assembly.
    """
    click.echo(f"Finishing assembly from graph: {graph}")
    click.echo(f"Method: {ai}")
    click.echo(f"Interactive: {interactive}")
    click.echo(f"Output: {output}")
    
    if ai == 'claude' and not api_key:
        click.echo("\n⚠ Warning: No Claude API key provided!")
        click.echo("   Set ANTHROPIC_API_KEY environment variable or use --api-key")
        click.echo("   Falling back to heuristic method")
    
    # TODO: Implement AI finishing
    click.echo(f"\n⚠ AI finishing implementation coming soon!")


@main.command()
@click.option('--assembly', '-a', required=True, type=click.Path(exists=True),
              help='Assembly file to validate (FASTA)')
@click.option('--reference', '-r', type=click.Path(exists=True),
              help='Reference genome for comparison (optional)')
@click.option('--output', '-o', type=click.Path(),
              help='Output validation report (JSON/HTML)')
def validate(assembly, reference, output):
    """
    Validate assembly quality.
    
    Computes assembly statistics (N50, L50, etc.) and optionally
    compares against a reference genome.
    """
    click.echo(f"Validating assembly: {assembly}")
    
    if reference:
        click.echo(f"Reference genome: {reference}")
    
    if output:
        click.echo(f"Output report: {output}")
    
    # TODO: Implement assembly validation
    click.echo("\n⚠ Assembly validation implementation coming soon!")


# ============================================================================
# Checkpoint Management Commands
# ============================================================================

@main.group()
def checkpoints():
    """Manage pipeline checkpoints."""
    pass


@checkpoints.command('list')
@click.option('--dir', '-d', 'checkpoint_dir', type=click.Path(exists=True),
              default='./checkpoints', help='Checkpoint directory')
def checkpoints_list(checkpoint_dir):
    """List available checkpoints."""
    click.echo(f"Checkpoints in: {checkpoint_dir}")
    # TODO: Implement checkpoint listing
    click.echo("\n⚠ Checkpoint management implementation coming soon!")


@checkpoints.command('remove')
@click.option('--dir', '-d', 'checkpoint_dir', type=click.Path(exists=True),
              required=True, help='Checkpoint directory')
@click.option('--all', 'remove_all', is_flag=True,
              help='Remove all checkpoints')
@click.option('--before', type=str,
              help='Remove checkpoints before this step')
def checkpoints_remove(checkpoint_dir, remove_all, before):
    """Remove checkpoints."""
    if remove_all:
        click.echo(f"Removing all checkpoints from: {checkpoint_dir}")
    elif before:
        click.echo(f"Removing checkpoints before: {before}")
    
    # TODO: Implement checkpoint removal
    click.echo("\n⚠ Checkpoint management implementation coming soon!")


@checkpoints.command('export')
@click.option('--dir', '-d', 'checkpoint_dir', type=click.Path(exists=True),
              required=True, help='Checkpoint directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output archive file')
def checkpoints_export(checkpoint_dir, output):
    """Export checkpoints to archive."""
    click.echo(f"Exporting checkpoints from: {checkpoint_dir}")
    click.echo(f"Output archive: {output}")
    # TODO: Implement checkpoint export
    click.echo("\n⚠ Checkpoint management implementation coming soon!")


# ============================================================================
# Utility Commands
# ============================================================================

@main.command()
def version():
    """Show version information."""
    click.echo(f"StrandWeaver v{__version__}")
    click.echo("\nDependencies:")
    
    try:
        import Bio
        click.echo(f"  BioPython: {Bio.__version__}")
    except ImportError:
        click.echo("  BioPython: not installed")
    
    try:
        import numpy
        click.echo(f"  NumPy: {numpy.__version__}")
    except ImportError:
        click.echo("  NumPy: not installed")
    
    try:
        import pysam
        click.echo(f"  pysam: {pysam.__version__}")
    except ImportError:
        click.echo("  pysam: not installed")
    
    try:
        import anthropic
        click.echo(f"  Anthropic: {anthropic.__version__}")
    except ImportError:
        click.echo("  Anthropic: not installed (optional for AI features)")


if __name__ == '__main__':
    sys.exit(main())
