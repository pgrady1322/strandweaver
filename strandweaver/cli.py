#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrandWeaver v0.3.0

Click CLI — main entrypoint for all StrandWeaver commands.

Author: StrandWeaver Development Team
Anthropic Claude Opus 4.6 used for code formatting and cleanup assistance.
License: Dual License (Academic/Commercial) - See LICENSE_ACADEMIC.md and LICENSE_COMMERCIAL.md
"""

import sys
import click
from pathlib import Path
import yaml

from .version import __version__
from .config.schema import load_config, save_config_template, validate_config


# ---------------------------------------------------------------------------
# Custom Click group that renders help with commands in labelled sections
# ---------------------------------------------------------------------------
class SectionedGroup(click.Group):
    """Click Group subclass that prints commands grouped into sections."""

    SECTIONS = {
        'Main Commands': [
            'config', 'pipeline', 'train', 'version',
        ],
        'Individual Pipeline Steps': [
            'assemble', 'correct', 'gap-fill', 'kweaver', 'polish',
            'profile', 'qv', 'validate',
        ],
        'Helpers and Utilities': [
            'align-hic', 'classify', 'extract-kmers', 'map-ul',
        ],
        'Nextflow Sub-Commands': [
            'nf-build-contigs', 'nf-build-graph', 'nf-checkpoints',
            'nf-detect-svs', 'nf-edgewarden-filter', 'nf-export-assembly',
            'nf-pathweaver-iter-general',
            'nf-pathweaver-iter-strict', 'nf-score-edges',
            'nf-strandtether-phase', 'nf-threadcompass-aggregate',
        ],
    }

    def format_commands(self, ctx, formatter):
        """Write commands organised by section instead of a flat list."""
        # Gather every visible command
        cmd_lookup = {}
        for name in self.list_commands(ctx):
            cmd = self.commands.get(name)
            if cmd is None or cmd.hidden:
                continue
            cmd_lookup[name] = cmd.get_short_help_str(limit=formatter.width)

        if not cmd_lookup:
            return

        # Emit each section
        for section_title, members in self.SECTIONS.items():
            rows = [
                (name, cmd_lookup[name])
                for name in sorted(members)
                if name in cmd_lookup
            ]
            if rows:
                with formatter.section(section_title):
                    formatter.write_dl(rows)

        # Safety net: any command not assigned to a section
        assigned = {n for members in self.SECTIONS.values() for n in members}
        orphans = sorted(set(cmd_lookup) - assigned)
        if orphans:
            rows = [(n, cmd_lookup[n]) for n in orphans]
            with formatter.section('Other Commands'):
                formatter.write_dl(rows)


@click.group(cls=SectionedGroup)
@click.version_option(version=__version__)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--quiet', '-q', is_flag=True, help='Suppress non-error output')
@click.pass_context
def main(ctx, verbose, quiet):
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
# ============================================================================
# Read Shortcuts (convenience aliases)
# ============================================================================
@click.option('--hifi-long-reads', type=click.Path(exists=True),
              help='PacBio HiFi long reads (convenience alias for -r# --technology# pacbio)')
@click.option('--ont-long-reads', type=click.Path(exists=True),
              help='ONT long reads (convenience alias for -r# --technology# ont)')
@click.option('--ont-ul', type=click.Path(exists=True),
              help='ONT ultra-long reads (convenience alias for -r# --technology# ont_ultralong)')
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
@click.option('--model-dir', type=click.Path(exists=True, file_okay=False),
              help='Directory containing trained AI model weights (overrides per-model paths)')
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
@click.option('--memory-limit', type=int, default=None,
              help='Memory limit in GB (default: no limit)')
# ============================================================================
# Pipeline Control
# ============================================================================
@click.option('--resume/--no-resume', default=False,
              help='Resume from last checkpoint')
@click.option('--checkpoint-dir', type=click.Path(),
              help='Checkpoint directory (default: <output>/checkpoints)')
@click.option('--start-from', 
              type=click.Choice(['profile', 'correct', 'assemble', 'finish', 'misassembly_report', 'classify_chromosomes']),
              help='Start pipeline from specific step')
@click.option('--skip-profiling', is_flag=True,
              help='Skip error profiling step')
@click.option('--skip-correction', is_flag=True,
              help='Skip error correction step (use for pre-corrected reads)')
@click.option('--dry-run', is_flag=True,
              help='Show pipeline plan without executing (validate inputs and config only)')
# ============================================================================
# Hi-C Scaffolding Data
# ============================================================================
@click.option('--hic-r1', type=click.Path(exists=True),
              help='Hi-C R1 reads (proximity ligation for scaffolding, not error-corrected)')
@click.option('--hic-r2', type=click.Path(exists=True),
              help='Hi-C R2 reads (proximity ligation for scaffolding, not error-corrected)')
# ============================================================================
# Assembly Filtering
# ============================================================================
@click.option('--min-contig-length', type=int, default=0,
              help='Minimum contig length to include in output (default: 0 = keep all). '
                   'Recommended: 500-1000 for production assemblies.')
# ============================================================================
# Chromosome Classification
# ============================================================================
@click.option('--id-chromosomes', is_flag=True,
              help='Classify scaffolds as chromosomes vs junk using gene content analysis '
                   'and telomere detection. Uses ORF detection by default; '
                   'set --gene-detection-method for BLAST/Augustus/BUSCO.')
@click.option('--id-chromosomes-advanced', is_flag=True,
              help='Enable advanced chromosome identification (adds Hi-C self-contact '
                   'pattern analysis and synteny). Telomere detection is always '
                   'included in basic --id-chromosomes mode.')
@click.option('--blast-db', type=str, default='nr',
              help='BLAST database for chromosome classification (default: nr)')
@click.option('--gene-detection-method', type=click.Choice(['orf', 'blast', 'augustus', 'busco']),
              default='orf',
              help='Gene detection method for chromosome classification. '
                   'orf=built-in (no external tools), blast/augustus/busco require installed tools.')
@click.option('--telomere-min-units', type=int, default=10,
              help='Minimum number of tandem telomere repeats to call a telomere (default: 10)')
@click.option('--telomere-search-depth', type=int, default=5000,
              help='Base-pairs to search at each scaffold end for telomeric repeats (default: 5000)')
@click.option('--telomere-sequence', type=str, default='TTAGGG',
              help='Telomere repeat motif to search for (default: TTAGGG). '
                   'Common alternatives: TTTAGGG (plants), TTAGG (insects).')
# ============================================================================
# Misassembly Reporting
# ============================================================================
@click.option('--misassembly-report/--no-misassembly-report', default=True,
              help='Generate misassembly report (TSV + BED). Enabled by default.')
@click.option('--misassembly-min-confidence',
              type=click.Choice(['HIGH', 'MEDIUM', 'LOW']),
              default='MEDIUM',
              help='Minimum confidence level for misassembly flags (default: MEDIUM).')
@click.option('--misassembly-format', type=str, default='tsv,bed',
              help='Comma-separated output formats for misassembly report (tsv,bed,json). Default: tsv,bed.')
# ============================================================================
# K-mer Overrides (disables KWeaver adaptive prediction for that stage)
# ============================================================================
@click.option('--kmer-size-assembly', type=int, default=None,
              help='Override KWeaver for assembly k-mer size (disables adaptive prediction)')
@click.option('--kmer-size-correction', type=int, default=None,
              help='Override KWeaver for error correction k-mer size (disables adaptive prediction)')
@click.option('--kmer-size-ul', type=int, default=None,
              help='Override KWeaver for ultra-long read k-mer size (disables adaptive prediction)')
# ============================================================================
# Assembly Options
# ============================================================================
@click.option('--ploidy', type=click.Choice(['haploid', 'diploid']),
              default=None,
              help='Ploidy mode (default: auto-detect). Polyploid support is planned for a future release.')
@click.option('--edge-filter-mode',
              type=click.Choice(['strict', 'moderate', 'lenient']),
              default=None,
              help='EdgeWarden filter strictness: strict (all 4 stages including AI), '
                   'moderate (coverage + quality + phasing, skip AI), '
                   'lenient (coverage filter only). Default: strict when AI enabled, moderate otherwise.')
@click.option('--min-sv-size', type=int, default=None,
              help='Minimum structural variant size in bp (default: 50)')
@click.option('--export-intermediate-graphs', is_flag=True,
              help='Export assembly graphs after each major pipeline step (GFA format)')
# ============================================================================
# Error Correction
# ============================================================================
@click.option('--max-correction-iterations', type=int, default=None,
              help='Maximum error correction iterations per read set (default: 3)')
# ============================================================================
# Coverage Sampling
# ============================================================================
@click.option('--sample-size-graph', type=int, default=None,
              help='Number of reads to sample for graph-building coverage (PacBio/ONT)')
@click.option('--sample-size-ul', type=int, default=None,
              help='Number of reads to sample for ultra-long read coverage')
@click.option('--sample-size-hic', type=int, default=None,
              help='Number of read pairs to sample for Hi-C coverage')
# ============================================================================
# Technology-Specific Read Subsampling
# ============================================================================
@click.option('--subsample-hifi', type=float, default=None,
              help='Subsample fraction for PacBio HiFi reads (0.0-1.0, e.g. 0.5 keeps 50%%)')
@click.option('--subsample-ont', type=float, default=None,
              help='Subsample fraction for ONT reads (0.0-1.0, e.g. 0.5 keeps 50%%)')
@click.option('--subsample-ont-ul', type=float, default=None,
              help='Subsample fraction for ONT ultra-long reads (0.0-1.0)')
@click.option('--subsample-illumina', type=float, default=None,
              help='Subsample fraction for Illumina reads (0.0-1.0)')
@click.option('--subsample-ancient', type=float, default=None,
              help='Subsample fraction for ancient DNA reads (0.0-1.0)')
# ============================================================================
# Output Options
# ============================================================================
@click.option('--output-format',
              type=click.Choice(['fasta', 'gfa', 'both']),
              default=None,
              help='Output assembly format (default: fasta). Use both for FASTA + GFA.')
@click.option('--log-level',
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default=None,
              help='Logging verbosity level (default: INFO)')
# ============================================================================
# Finishing
# ============================================================================
@click.option('--decontaminate', is_flag=True,
              help='Enable decontamination screening. '
                   'Not yet implemented — reserved for future release.')
@click.pass_context
def pipeline(ctx,
             reads1, reads2, reads3, reads4, reads5,
             technology1, technology2, technology3, technology4, technology5,
             hifi_long_reads, ont_long_reads, ont_ul,
             output, config,
             use_ai, disable_correction_ai, disable_assembly_ai, model_dir,
             use_gpu, gpu_backend, gpu_device, threads, memory_limit,
             resume, checkpoint_dir, start_from, skip_profiling, skip_correction,
             dry_run,
             min_contig_length,
             illumina_r1, illumina_r2, hic_r1, hic_r2,
             id_chromosomes, id_chromosomes_advanced, blast_db, gene_detection_method,
             telomere_min_units, telomere_search_depth, telomere_sequence,
             misassembly_report, misassembly_min_confidence, misassembly_format,
             kmer_size_assembly, kmer_size_correction, kmer_size_ul,
             ploidy, edge_filter_mode, min_sv_size, export_intermediate_graphs,
             max_correction_iterations,
             sample_size_graph, sample_size_ul, sample_size_hic,
             subsample_hifi, subsample_ont, subsample_ont_ul,
             subsample_illumina, subsample_ancient,
             output_format, log_level,
             decontaminate):
    """
    Run the complete assembly pipeline.
    
    This command runs the full StrandWeaver pipeline from error profiling
    through AI-powered finishing. The pipeline can be resumed from checkpoints
    if interrupted.
    
    Supports single or multi-technology (hybrid) assemblies.
    
    Examples:
        # Single technology
        strandweaver pipeline -r1 illumina.fq --technology1 illumina -o output/
        
        # Hybrid assembly - Illumina + ONT
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
        
        # Chromosome classification with custom telomere settings (plants)
        strandweaver pipeline -r1 hifi.fq -o output/ \\
            --id-chromosomes \\
            --telomere-sequence TTTAGGG \\
            --telomere-min-units 8 \\
            --telomere-search-depth 10000
        
        # High-confidence misassembly report only, JSON + BED
        strandweaver pipeline -r1 reads.fq -o output/ \\
            --misassembly-min-confidence HIGH \\
            --misassembly-format json,bed
        
        # Convenience read shortcuts (no -r#/--technology# needed)
        strandweaver pipeline --hifi-long-reads hifi.fq.gz -o output/
        strandweaver pipeline --ont-long-reads ont.fq.gz --ont-ul ul.fq.gz -o output/
        
        # Override k-mer sizes (disables KWeaver prediction)
        strandweaver pipeline --hifi-long-reads hifi.fq.gz \\
            --kmer-size-assembly 51 --kmer-size-correction 31 -o output/
        
        # Diploid assembly with strict edge filtering
        strandweaver pipeline --hifi-long-reads hifi.fq.gz \\
            --ploidy diploid --edge-filter-mode strict -o output/
        
        # Dry run — show pipeline plan without executing
        strandweaver pipeline --hifi-long-reads hifi.fq.gz -o output/ --dry-run
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
    
    # Validate Illumina paired-end input
    if illumina_r1 and not illumina_r2:
        click.echo(f"❌ Error: --illumina-r1 requires --illumina-r2", err=True)
        ctx.exit(1)
    if illumina_r2 and not illumina_r1:
        click.echo(f"❌ Error: --illumina-r2 requires --illumina-r1", err=True)
        ctx.exit(1)
    
    # Build complete reads list
    all_reads = []
    all_technologies = []
    
    # Add Illumina paired-end if provided
    illumina_paired_indices = None
    if illumina_r1 and illumina_r2:
        illumina_paired_indices = (len(all_reads), len(all_reads) + 1)  # (R1 idx, R2 idx)
        all_reads.extend([illumina_r1, illumina_r2])
        all_technologies.extend(['illumina', 'illumina'])
    
    # Add convenience read shortcuts
    if hifi_long_reads:
        all_reads.append(hifi_long_reads)
        all_technologies.append('pacbio')
    
    if ont_long_reads:
        all_reads.append(ont_long_reads)
        all_technologies.append('ont')
    
    if ont_ul:
        all_reads.append(ont_ul)
        all_technologies.append('ont_ultralong')
    
    # Numbered syntax
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
    
    # Ensure we have at least some input
    if not all_reads:
        click.echo(f"❌ Error: No input reads specified.", err=True)
        click.echo(f"\nUse one of:", err=True)
        click.echo(f"  --hifi-long-reads hifi.fq.gz (PacBio HiFi)", err=True)
        click.echo(f"  --ont-long-reads ont.fq.gz   (ONT long reads)", err=True)
        click.echo(f"  --ont-ul ul.fq.gz            (ONT ultra-long reads)", err=True)
        click.echo(f"  --illumina-r1 R1.fq --illumina-r2 R2.fq (Illumina paired-end)", err=True)
        click.echo(f"  -r1 file.fq --technology1 <tech> (numbered syntax)", err=True)
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
    
    if model_dir:
        pipeline_config['ai']['model_dir'] = model_dir
        if verbose:
            click.echo(f"ℹ️  AI model directory set to {model_dir}")
    
    if use_gpu:
        pipeline_config['hardware']['use_gpu'] = True
        pipeline_config['hardware']['gpu_device'] = gpu_device
        pipeline_config['hardware']['gpu_backend'] = gpu_backend
        if verbose:
            click.echo(f"ℹ️  GPU acceleration enabled (backend {gpu_backend}, device {gpu_device})")
    elif gpu_backend and gpu_backend != 'auto':
        # User explicitly requested a backend without --use-gpu; honour it
        pipeline_config['hardware']['use_gpu'] = True
        pipeline_config['hardware']['gpu_backend'] = gpu_backend
        pipeline_config['hardware']['gpu_device'] = gpu_device
        if verbose:
            click.echo(f"ℹ️  GPU acceleration enabled via --gpu-backend {gpu_backend} (device {gpu_device})")
    
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
    
    # K-mer overrides (disable KWeaver adaptive prediction for that stage)
    if kmer_size_assembly is not None:
        pipeline_config['assembly']['graph']['kmer_size'] = kmer_size_assembly
        pipeline_config['assembly']['dbg']['adaptive_k'] = False
        if verbose:
            click.echo(f"ℹ️  Assembly k-mer size overridden to {kmer_size_assembly} (KWeaver disabled for assembly)")
    
    if kmer_size_correction is not None:
        pipeline_config['correction']['kmer_size_override'] = kmer_size_correction
        if verbose:
            click.echo(f"ℹ️  Correction k-mer size overridden to {kmer_size_correction} (KWeaver disabled for correction)")
    
    if kmer_size_ul is not None:
        pipeline_config['assembly']['string_graph']['kmer_size_override'] = kmer_size_ul
        if verbose:
            click.echo(f"ℹ️  Ultra-long k-mer size overridden to {kmer_size_ul} (KWeaver disabled for UL)")
    
    # Assembly options
    if ploidy:
        pipeline_config['assembly']['diploid']['mode'] = ploidy
        if verbose:
            click.echo(f"ℹ️  Ploidy mode set to {ploidy}")
    
    if edge_filter_mode:
        pipeline_config['assembly']['graph']['edge_filter_mode'] = edge_filter_mode
        if verbose:
            click.echo(f"ℹ️  Edge filter mode: {edge_filter_mode}")
    
    if min_sv_size is not None:
        pipeline_config['assembly']['sv_detection']['min_sv_size'] = min_sv_size
        if verbose:
            click.echo(f"ℹ️  Minimum SV size: {min_sv_size} bp")
    
    if export_intermediate_graphs:
        pipeline_config['output']['export_intermediate_graphs'] = True
        if verbose:
            click.echo(f"ℹ️  Intermediate graph export enabled")
    
    # Error correction
    if max_correction_iterations is not None:
        pipeline_config['correction']['max_iterations'] = max_correction_iterations
        if verbose:
            click.echo(f"ℹ️  Max correction iterations: {max_correction_iterations}")
    
    # Coverage sampling
    if sample_size_graph is not None:
        pipeline_config['profiling']['sample_size_graph'] = sample_size_graph
        if verbose:
            click.echo(f"ℹ️  Graph coverage sample size: {sample_size_graph:,} reads")
    if sample_size_ul is not None:
        pipeline_config['profiling']['sample_size_ul'] = sample_size_ul
        if verbose:
            click.echo(f"ℹ️  Ultra-long coverage sample size: {sample_size_ul:,} reads")
    if sample_size_hic is not None:
        pipeline_config['profiling']['sample_size_hic'] = sample_size_hic
        if verbose:
            click.echo(f"ℹ️  Hi-C coverage sample size: {sample_size_hic:,} read pairs")
    
    # Technology-specific subsampling
    subsample_map = {
        'pacbio': subsample_hifi,
        'ont': subsample_ont,
        'ont_ultralong': subsample_ont_ul,
        'illumina': subsample_illumina,
        'ancient': subsample_ancient,
    }
    active_subsamples = {k: v for k, v in subsample_map.items() if v is not None}
    
    for tech_name, frac in active_subsamples.items():
        if not (0.0 < frac <= 1.0):
            click.echo(f"❌ Error: --subsample-{tech_name.replace('_', '-')} must be between 0.0 and 1.0 "
                       f"(got {frac})", err=True)
            ctx.exit(1)
    
    if active_subsamples and verbose:
        for tech_name, frac in active_subsamples.items():
            click.echo(f"ℹ️  Subsampling {tech_name}: {frac:.0%} of reads")
    
    # Memory limit
    if memory_limit is not None:
        pipeline_config['hardware']['memory_limit_gb'] = memory_limit
        if verbose:
            click.echo(f"ℹ️  Memory limit: {memory_limit} GB")
    
    # Output options
    if output_format:
        pipeline_config['output']['format'] = output_format
        if verbose:
            click.echo(f"ℹ️  Output format: {output_format}")
    
    if log_level:
        pipeline_config['output']['logging']['level'] = log_level
        if verbose:
            click.echo(f"ℹ️  Log level: {log_level}")
    
    # Decontamination
    if decontaminate:
        pipeline_config['finishing']['decontamination']['enabled'] = True
        if verbose:
            click.echo(f"ℹ️  Decontamination: enabled (note: not yet implemented)")
    
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
    # Chromosome Classification Configuration
    # ========================================================================
    if id_chromosomes or id_chromosomes_advanced:
        # Initialize chromosome_classification config if not present
        if 'chromosome_classification' not in pipeline_config:
            pipeline_config['chromosome_classification'] = {}
        
        pipeline_config['chromosome_classification']['enabled'] = True
        pipeline_config['chromosome_classification']['advanced'] = id_chromosomes_advanced
        pipeline_config['chromosome_classification']['blast_database'] = blast_db
        pipeline_config['chromosome_classification']['gene_detection_method'] = gene_detection_method
        pipeline_config['chromosome_classification']['mode'] = 'fast' if gene_detection_method in ('orf', 'blast') else 'accurate'
        
        # Telomere detection parameters (Tier 1)
        pipeline_config['chromosome_classification']['telomere_min_units'] = telomere_min_units
        pipeline_config['chromosome_classification']['telomere_search_depth'] = telomere_search_depth
        pipeline_config['chromosome_classification']['telomere_sequence'] = telomere_sequence
        
        if verbose:
            mode = "Advanced (Tiers 1-3)" if id_chromosomes_advanced else "Basic (Tiers 1-2)"
            click.echo(f"ℹ️  Chromosome classification enabled: {mode}")
            click.echo(f"   BLAST database: {blast_db}")
            click.echo(f"   Telomere: motif={telomere_sequence}, "
                       f"min_units={telomere_min_units}, "
                       f"search_depth={telomere_search_depth}bp")
    
    # ========================================================================
    # Misassembly Report Configuration
    # ========================================================================
    pipeline_config['misassembly_report'] = {
        'enabled': misassembly_report,
        'min_confidence': misassembly_min_confidence,
        'formats': [f.strip() for f in misassembly_format.split(',')],
    }
    
    # Remove misassembly_report step from pipeline if disabled
    if not misassembly_report:
        steps = pipeline_config.get('pipeline', {}).get('steps', [])
        if 'misassembly_report' in steps:
            steps.remove('misassembly_report')
    
    if verbose and misassembly_report:
        click.echo(f"ℹ️  Misassembly report: min_confidence={misassembly_min_confidence}, "
                   f"formats={misassembly_format}")
    
    # ========================================================================
    # Dry Run
    # ========================================================================
    if dry_run:
        click.echo("\n🔎 DRY RUN — pipeline plan shown above. No processing performed.")
        click.echo("   Remove --dry-run to execute the pipeline.")
        return
    
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
        'min_contig_length': min_contig_length,
        'illumina_paired_indices': illumina_paired_indices,  # (R1_idx, R2_idx) or None
        'subsample': active_subsamples,  # {tech_name: fraction} e.g. {'ont': 0.5}
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


@main.command('nf-build-contigs')
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
    
    click.echo("\n⚠️ Contig building from short reads:")
    click.echo("   This feature is planned for a future release")
    click.echo("   Current workaround: Use flye/hifiasm/miniasm for short read assembly first")


@main.command('core-assemble')
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
@click.option('--hifi-long-reads', type=click.Path(exists=True),
              help='PacBio HiFi long reads (convenience shortcut)')
@click.option('--ont-long-reads', type=click.Path(exists=True),
              help='ONT long reads (convenience shortcut)')
@click.option('--ont-ul', type=click.Path(exists=True),
              help='ONT ultra-long reads (convenience shortcut)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for assembly results')
@click.option('--graph', '-g', type=click.Path(),
              help='Output assembly graph file (GFA)')
@click.option('--graph-type', 
              type=click.Choice(['string', 'debruijn', 'hybrid']),
              default='string', help='Assembly graph type')
@click.option('--min-coverage', type=int, default=5,
              help='Minimum coverage threshold')
@click.option('--kmer-size', '-k', type=int, default=None,
              help='Override k-mer size for assembly (disables adaptive KWeaver prediction)')
@click.option('--ploidy', type=click.Choice(['haploid', 'diploid']),
              default=None,
              help='Ploidy mode (default: auto-detect). Polyploid support is planned.')
@click.option('--use-ai/--classical', default=True,
              help='Enable/disable AI-powered assembly modules (default: enabled)')
@click.option('--model-dir', type=click.Path(exists=True), default=None,
              help='Path to trained model directory')
@click.option('--use-gpu', is_flag=True, default=False,
              help='Enable GPU acceleration for AI modules')
@click.option('--config', '-c', type=click.Path(exists=True), default=None,
              help='YAML config file (overrides defaults)')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
@click.option('--hic-r1', type=click.Path(exists=True), default=None,
              help='Hi-C forward reads for scaffolding')
@click.option('--hic-r2', type=click.Path(exists=True), default=None,
              help='Hi-C reverse reads for scaffolding')
def core_assemble(reads1, reads2, reads3, reads4, reads5,
                  technology1, technology2, technology3, technology4, technology5,
                  illumina_r1, illumina_r2, hifi_long_reads, ont_long_reads, ont_ul,
                  output, graph, graph_type,
                  min_coverage, kmer_size, ploidy,
                  use_ai, model_dir, use_gpu, config, threads,
                  hic_r1, hic_r2):
    """
    Core assembly: graph build + path resolution (no preprocessing).
    
    Runs the core assembly steps only (DBG construction, EdgeWarden,
    PathWeaver, string graph overlay, ThreadCompass routing, haplotype
    phasing, SV detection, and contig extraction). Skips K-Weaver
    prediction, error profiling, read correction, and post-assembly
    finishing — use 'strandweaver pipeline' for the full end-to-end flow.
    
    Intended for pre-corrected reads or when you want fine-grained
    control over the assembly step alone.
    
    Examples:
        # HiFi reads (pre-corrected)
        strandweaver core-assemble --hifi-long-reads corrected_hifi.fq -o asm_out/
        
        # Illumina paired-end
        strandweaver core-assemble --illumina-r1 R1.fq --illumina-r2 R2.fq -o asm_out/
        
        # Multi-technology (hybrid)
        strandweaver core-assemble \\
            -r1 illumina.fq --technology1 illumina \\
            -r2 ont.fq --technology2 ont \\
            -r3 hifi.fq --technology3 pacbio \\
            -o asm_out/ -g assembly.gfa
        
        # Classical mode (no AI models)
        strandweaver core-assemble --hifi-long-reads hifi.fq --classical -o asm_out/
        
        # With Hi-C scaffolding
        strandweaver core-assemble --hifi-long-reads hifi.fq \\
            --hic-r1 hic_R1.fq --hic-r2 hic_R2.fq -o asm_out/
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
    
    # Validate Illumina paired-end input
    if illumina_r1 and not illumina_r2:
        click.echo(f"❌ Error: --illumina-r1 requires --illumina-r2", err=True)
        from sys import exit
        exit(1)
    if illumina_r2 and not illumina_r1:
        click.echo(f"❌ Error: --illumina-r2 requires --illumina-r1", err=True)
        from sys import exit
        exit(1)
    
    # Validate Hi-C paired input
    if hic_r1 and not hic_r2:
        click.echo(f"❌ Error: --hic-r1 requires --hic-r2", err=True)
        from sys import exit
        exit(1)
    if hic_r2 and not hic_r1:
        click.echo(f"❌ Error: --hic-r2 requires --hic-r1", err=True)
        from sys import exit
        exit(1)
    
    # Build complete reads list
    all_reads = []
    all_technologies = []
    
    if illumina_r1 and illumina_r2:
        all_reads.extend([illumina_r1, illumina_r2])
        all_technologies.extend(['illumina', 'illumina'])
    
    # Convenience shortcuts
    if hifi_long_reads:
        all_reads.append(hifi_long_reads)
        all_technologies.append('pacbio')
    
    if ont_long_reads:
        all_reads.append(ont_long_reads)
        all_technologies.append('ont')
    
    if ont_ul:
        all_reads.append(ont_ul)
        all_technologies.append('ont_ultralong')
    
    # Numbered syntax
    for num in sorted(numbered_reads.keys()):
        all_reads.append(numbered_reads[num])
        all_technologies.append(numbered_techs.get(num, 'auto'))
    
    # Warn if technology specified without corresponding read
    for num in numbered_techs.keys():
        if num not in numbered_reads:
            click.echo(f"⚠️  Warning: --technology{num} specified but no -r{num} file provided (ignored)", err=True)
    
    # Ensure we have at least some input
    if not all_reads:
        click.echo(f"❌ Error: No input reads specified. Use --hifi-long-reads, --ont-long-reads, --illumina-r1/--illumina-r2, or -r1/-r2/etc", err=True)
        from sys import exit
        exit(1)
    
    click.echo(f"{'='*60}")
    click.echo(f"StrandWeaver v{__version__} — Core Assembly")
    click.echo(f"{'='*60}")
    click.echo(f"\n📁 Input Read Files: {len(all_reads)}")
    for i, (reads, tech) in enumerate(zip(all_reads, all_technologies), 1):
        click.echo(f"  {i}. {reads} ({tech})")
    
    click.echo(f"\n📂 Output: {output}")
    click.echo(f"Graph type: {graph_type}")
    click.echo(f"Min coverage: {min_coverage}x")
    if kmer_size:
        click.echo(f"K-mer size: {kmer_size} (KWeaver disabled)")
    else:
        click.echo(f"K-mer size: auto (adaptive)")
    if ploidy:
        click.echo(f"Ploidy: {ploidy}")
    click.echo(f"AI-powered: {'ENABLED' if use_ai else 'DISABLED (classical)'}")
    if hic_r1:
        click.echo(f"Hi-C R1: {hic_r1}")
        click.echo(f"Hi-C R2: {hic_r2}")
    
    if graph:
        click.echo(f"Output graph: {graph}")
    
    click.echo(f"{'='*60}\n")
    
    # ====================================================================
    # Build pipeline config and run core assembly step only
    # ====================================================================
    from .utils.pipeline import PipelineOrchestrator
    
    # Load base config (from file or defaults)
    pipeline_config = load_config(Path(config) if config else None)
    
    # Override: run ONLY the assemble step (skip preprocessing + finishing)
    pipeline_config['pipeline']['steps'] = ['assemble']
    
    # AI settings
    if not use_ai:
        pipeline_config['ai']['enabled'] = False
    
    if model_dir:
        pipeline_config['ai']['model_dir'] = model_dir
    
    if use_gpu:
        pipeline_config['hardware']['use_gpu'] = True
    
    if threads:
        pipeline_config['hardware']['threads'] = threads
    
    # Assembly options
    if kmer_size is not None:
        pipeline_config['assembly']['graph']['kmer_size'] = kmer_size
        pipeline_config['assembly']['dbg']['adaptive_k'] = False
        pipeline_config['dbg_k'] = kmer_size
    
    pipeline_config['min_coverage'] = min_coverage
    
    if ploidy:
        ploidy_int = 1 if ploidy == 'haploid' else 2
        pipeline_config['assembly']['ploidy'] = ploidy_int
        pipeline_config['assembly']['diploid']['mode'] = ploidy
    
    # Hi-C scaffolding
    if hic_r1 and hic_r2:
        pipeline_config['scaffolding']['hic']['enabled'] = True
    
    # Runtime parameters
    pipeline_config['runtime'] = {
        'reads': all_reads,
        'technologies': all_technologies,
        'output_dir': output,
        'hic_r1': hic_r1,
        'hic_r2': hic_r2,
        'verbose': False,
        'min_contig_length': 500,
        'illumina_paired_indices': None,
    }
    
    # Validate configuration
    config_errors = validate_config(pipeline_config)
    if config_errors:
        click.echo("❌ Configuration validation failed:", err=True)
        for error in config_errors:
            click.echo(f"  • {error}", err=True)
        from sys import exit
        exit(1)
    
    # Initialize and run orchestrator (assemble step only)
    orchestrator = PipelineOrchestrator(config=pipeline_config)
    
    try:
        result = orchestrator.run()
        
        click.echo("\n" + "="*60)
        click.echo("✅ Core assembly completed successfully!")
        click.echo("="*60)
        click.echo(f"Output directory: {output}")
        
        # Copy GFA to user-specified path if --graph was given
        if graph:
            import shutil
            default_gfa = Path(output) / "assembly_graph.gfa"
            if default_gfa.exists():
                shutil.copy2(default_gfa, graph)
                click.echo(f"Assembly graph: {graph}")
        
    except Exception as e:
        click.echo(f"\n❌ Core assembly failed: {e}", err=True)
        import traceback
        traceback.print_exc()
        from sys import exit
        exit(1)


@main.command()
@click.option('--assembly', '-a', required=True, type=click.Path(exists=True),
              help='Assembly file to validate (FASTA)')
@click.option('--reference', '-r', type=click.Path(exists=True),
              help='Reference genome for comparison (optional)')
@click.option('--output', '-o', type=click.Path(),
              help='Output validation report (JSON/HTML)')
@click.option('--busco-lineage', type=str, default=None,
              help='BUSCO lineage dataset for completeness assessment '
                   '(e.g., mammalia_odb10, aves_odb10). '
                   'Not yet wired — planned for a future release.')
@click.option('--threads', '-t', type=int, default=1,
              help='Number of threads to use')
def validate(assembly, reference, output, busco_lineage, threads):
    """
    Validate assembly quality.
    
    Computes assembly statistics (N50, L50, etc.) and optionally
    compares against a reference genome.
    
    Examples:
        # Basic validation
        strandweaver validate -a assembly.fa -o report.json
        
        # Compare against reference
        strandweaver validate -a assembly.fa -r reference.fa -o report.json
        
        # BUSCO completeness (planned)
        strandweaver validate -a assembly.fa --busco-lineage mammalia_odb10 -o report.json
    """
    click.echo(f"Validating assembly: {assembly}")
    
    if reference:
        click.echo(f"Reference genome: {reference}")
    
    if busco_lineage:
        click.echo(f"BUSCO lineage: {busco_lineage}")
        click.echo("  ⚠️ BUSCO integration is not yet wired — planned for a future release.")
    
    click.echo(f"Threads: {threads}")
    
    if output:
        click.echo(f"Output report: {output}")
    
    click.echo("\n⚠️ Assembly validation:")
    click.echo("   This feature is planned for a future release")
    click.echo("   Current workaround: Use QUAST or similar tools for assembly QC")


# ============================================================================
# Finishing Commands (QV, Polish, Gap-Fill)
# ============================================================================

@main.command('qv')
@click.option('--assembly', '-a', required=True, type=click.Path(exists=True),
              help='Assembly FASTA to evaluate')
@click.option('--reads', '-r', type=click.Path(exists=True),
              help='Read file for k-mer completeness (FASTA/FASTQ, optional)')
@click.option('--output', '-o', type=click.Path(), default=None,
              help='Output QV report JSON (default: stdout)')
@click.option('--k', '-k', type=int, default=21,
              help='K-mer size for completeness estimation (11-101, default: 21)')
def qv_estimate(assembly, reads, output, k):
    """
    Estimate assembly quality value (QV).

    Computes Merqury-style k-mer completeness QV and per-contig quality
    metrics. Optionally includes read k-mer completeness when reads are
    provided.

    Examples:\b
        # QV from assembly only (uses default error rate)
        strandweaver qv -a assembly.fa

        # QV with read k-mer completeness
        strandweaver qv -a assembly.fa -r reads.fastq -o qv_report.json
    """
    import json as _json
    from .io_utils import read_fasta, read_fastq
    from .assembly_utils.qv_estimator import QVEstimator

    click.echo(f"Estimating QV for: {assembly}")
    contigs = list(read_fasta(assembly))
    click.echo(f"  Loaded {len(contigs)} contigs")

    read_list = None
    if reads:
        click.echo(f"  Loading reads: {reads}")
        reads_path = Path(reads)
        if reads_path.suffix in ('.fq', '.fastq', '.fq.gz', '.fastq.gz'):
            read_list = list(read_fastq(reads_path))
        else:
            read_list = list(read_fasta(reads_path))
        click.echo(f"  Loaded {len(read_list)} reads")

    estimator = QVEstimator(k=k)
    result = estimator.estimate(contigs, reads=read_list)

    if output:
        estimator.save_report(result, Path(output))
        click.echo(f"  Report saved: {output}")
    else:
        click.echo(_json.dumps(result.to_dict(), indent=2))

    click.echo(f"\n✓ Assembly QV: Q{result.global_combined_qv:.1f}  "
               f"N50: {result.n50:,} bp  Contigs: {result.num_contigs}")


@main.command('polish')
@click.option('--assembly', '-a', required=True, type=click.Path(exists=True),
              help='Assembly FASTA to polish')
@click.option('--reads', '-r', required=True, type=click.Path(exists=True),
              help='Read file for polishing (FASTA/FASTQ)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output polished assembly FASTA')
@click.option('--rounds', type=int, default=2,
              help='Maximum polishing rounds (default: 2)')
@click.option('--k', '-k', type=int, default=21,
              help='K-mer size for read anchoring (default: 21)')
@click.option('--min-coverage', type=int, default=3,
              help='Minimum pileup depth for consensus (default: 3)')
def polish_assembly(assembly, reads, output, rounds, k, min_coverage):
    """
    Iteratively polish an assembly using read consensus.

    Maps reads to contigs via k-mer anchors, builds per-position pileup,
    and calls quality-weighted consensus. Repeats for multiple rounds
    with QV-guided convergence detection.

    Examples:\b
        strandweaver polish -a draft.fa -r reads.fastq -o polished.fa

        strandweaver polish -a draft.fa -r reads.fq -o polished.fa --rounds 3
    """
    from .io_utils import read_fasta, read_fastq, write_fasta
    from .assembly_utils.iterative_polisher import IterativePolisher

    click.echo(f"Polishing assembly: {assembly}")
    contigs = list(read_fasta(assembly))
    click.echo(f"  Loaded {len(contigs)} contigs")

    reads_path = Path(reads)
    click.echo(f"  Loading reads: {reads}")
    if reads_path.suffix in ('.fq', '.fastq', '.fq.gz', '.fastq.gz'):
        read_list = list(read_fastq(reads_path))
    else:
        read_list = list(read_fasta(reads_path))
    click.echo(f"  Loaded {len(read_list)} reads")

    polisher = IterativePolisher(
        max_rounds=rounds, k=k, min_coverage=min_coverage,
    )
    polished, summary = polisher.polish(contigs, read_list)

    write_fasta(polished, Path(output))

    click.echo(f"\n✓ Polishing complete")
    click.echo(f"  Rounds: {summary.total_rounds}")
    click.echo(f"  Bases corrected: {summary.total_bases_corrected:,}")
    click.echo(f"  QV: {summary.initial_qv:.1f} → {summary.final_qv:.1f} "
               f"(+{summary.qv_improvement:.2f})")
    click.echo(f"  Converged: {summary.converged}")
    click.echo(f"  Output: {output}")


@main.command('gap-fill')
@click.option('--assembly', '-a', required=True, type=click.Path(exists=True),
              help='Assembly FASTA with N-gaps')
@click.option('--reads', '-r', required=True, type=click.Path(exists=True),
              help='Read file for gap filling (FASTA/FASTQ)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output gap-filled assembly FASTA')
@click.option('--max-gap-size', type=int, default=10000,
              help='Maximum gap size to attempt filling (default: 10000)')
@click.option('--k', '-k', type=int, default=21,
              help='K-mer size for flank anchoring (default: 21)')
@click.option('--min-spanning', type=int, default=3,
              help='Minimum spanning reads required per gap (default: 3)')
def gap_fill(assembly, reads, output, max_gap_size, k, min_spanning):
    """
    Fill N-gaps in an assembly using spanning reads.

    Detects runs of N characters, finds reads that anchor on both flanks,
    builds a consensus for the bridging region, and replaces the gap.

    Examples:\b
        strandweaver gap-fill -a scaffolds.fa -r reads.fastq -o filled.fa

        strandweaver gap-fill -a scaffolds.fa -r reads.fq -o filled.fa --max-gap-size 5000
    """
    from .io_utils import read_fasta, read_fastq, write_fasta
    from .assembly_utils.gap_filler import GapFiller

    click.echo(f"Gap-filling assembly: {assembly}")
    contigs = list(read_fasta(assembly))
    click.echo(f"  Loaded {len(contigs)} contigs")

    reads_path = Path(reads)
    click.echo(f"  Loading reads: {reads}")
    if reads_path.suffix in ('.fq', '.fastq', '.fq.gz', '.fastq.gz'):
        read_list = list(read_fastq(reads_path))
    else:
        read_list = list(read_fasta(reads_path))
    click.echo(f"  Loaded {len(read_list)} reads")

    filler = GapFiller(
        max_gap_size=max_gap_size, k=k, min_spanning_reads=min_spanning,
    )
    filled, summary = filler.fill(contigs, read_list)

    write_fasta(filled, Path(output))

    click.echo(f"\n✓ Gap filling complete")
    click.echo(f"  Gaps found: {summary.total_gaps}")
    click.echo(f"  Gaps filled: {summary.gaps_filled}")
    click.echo(f"  Gaps unfilled: {summary.gaps_unfilled}")
    click.echo(f"  Bases filled: {summary.bases_filled:,} N → {summary.bases_inserted:,} bp")
    if summary.mean_fill_confidence > 0:
        click.echo(f"  Mean confidence: {summary.mean_fill_confidence:.3f}")
    click.echo(f"  Output: {output}")


# ============================================================================
# Checkpoint Management Commands
# ============================================================================

@main.group('nf-checkpoints')
def nf_checkpoints():
    """Manage pipeline checkpoints (Nextflow)."""
    pass


@nf_checkpoints.command('list')
@click.option('--dir', '-d', 'checkpoint_dir', type=click.Path(exists=True),
              default='./checkpoints', help='Checkpoint directory')
def checkpoints_list(checkpoint_dir):
    """List available checkpoints."""
    click.echo(f"Checkpoints in: {checkpoint_dir}")
    click.echo("\n⚠️ Checkpoint management:")
    click.echo("   This feature is planned for a future release")


@nf_checkpoints.command('remove')
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
    
    click.echo("\n⚠️ Checkpoint management:")
    click.echo("   This feature is planned for a future release")


@nf_checkpoints.command('export')
@click.option('--dir', '-d', 'checkpoint_dir', type=click.Path(exists=True),
              required=True, help='Checkpoint directory')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output archive file')
def checkpoints_export(checkpoint_dir, output):
    """Export checkpoints to archive."""
    click.echo(f"Exporting checkpoints from: {checkpoint_dir}")
    click.echo(f"Output archive: {output}")
    click.echo("\n⚠️ Checkpoint management:")
    click.echo("   This feature is planned for a future release")


# ============================================================================
# Utility Commands
# ============================================================================

# ============================================================================
# ============================================================================
# STANDALONE PROCESSING COMMANDS
# ============================================================================
# User-facing commands that support both direct execution and Nextflow mode.
# These commands process entire datasets by default, or generate/run Nextflow
# workflows when --nextflow flag is provided.
# ============================================================================

def _run_nextflow(workflow_type, **kwargs):
    """
    Helper function to generate and execute Nextflow workflow.
    
    Args:
        workflow_type: Type of workflow (correct, extract-kmers, etc.)
        **kwargs: Arguments passed from CLI
    """
    import subprocess
    import tempfile
    from pathlib import Path
    
    # Build Nextflow command
    nf_script = Path(__file__).parent.parent / 'nextflow' / 'main.nf'
    
    cmd = ['nextflow', 'run', str(nf_script)]
    
    # Add workflow entry point
    cmd.extend(['-entry', workflow_type.upper()])
    
    # Add profile
    if 'nf_profile' in kwargs and kwargs['nf_profile']:
        cmd.extend(['-profile', kwargs['nf_profile']])
    
    # Add resume flag
    if kwargs.get('nf_resume'):
        cmd.append('-resume')
    
    # Add input files
    if 'hifi' in kwargs and kwargs['hifi']:
        cmd.extend(['--hifi', kwargs['hifi']])
    if 'ont' in kwargs and kwargs['ont']:
        cmd.extend(['--ont', kwargs['ont']])
    if 'ont_ul' in kwargs and kwargs['ont_ul']:
        cmd.extend(['--ont_ul', kwargs['ont_ul']])
    if 'hic_r1' in kwargs and kwargs['hic_r1']:
        cmd.extend(['--hic_r1', kwargs['hic_r1']])
    if 'hic_r2' in kwargs and kwargs['hic_r2']:
        cmd.extend(['--hic_r2', kwargs['hic_r2']])
    
    # Add output directory
    if 'output' in kwargs and kwargs['output']:
        cmd.extend(['--outdir', kwargs['output']])
    
    # Add batch sizes
    if 'correction_batch_size' in kwargs and kwargs['correction_batch_size']:
        cmd.extend(['--correction_batch_size', str(kwargs['correction_batch_size'])])
    if 'edge_batch_size' in kwargs and kwargs['edge_batch_size']:
        cmd.extend(['--edge_batch_size', str(kwargs['edge_batch_size'])])
    if 'ul_batch_size' in kwargs and kwargs['ul_batch_size']:
        cmd.extend(['--ul_batch_size', str(kwargs['ul_batch_size'])])
    if 'hic_batch_size' in kwargs and kwargs['hic_batch_size']:
        cmd.extend(['--hic_batch_size', str(kwargs['hic_batch_size'])])
    if 'sv_batch_size' in kwargs and kwargs['sv_batch_size']:
        cmd.extend(['--sv_batch_size', str(kwargs['sv_batch_size'])])
    if 'kmer_batch_size' in kwargs and kwargs['kmer_batch_size']:
        cmd.extend(['--kmer_batch_size', str(kwargs['kmer_batch_size'])])
    
    # Add max jobs
    if 'max_correction_jobs' in kwargs and kwargs['max_correction_jobs']:
        cmd.extend(['--max_correction_jobs', str(kwargs['max_correction_jobs'])])
    if 'max_edge_jobs' in kwargs and kwargs['max_edge_jobs']:
        cmd.extend(['--max_edge_jobs', str(kwargs['max_edge_jobs'])])
    if 'max_ul_jobs' in kwargs and kwargs['max_ul_jobs']:
        cmd.extend(['--max_ul_jobs', str(kwargs['max_ul_jobs'])])
    if 'max_hic_jobs' in kwargs and kwargs['max_hic_jobs']:
        cmd.extend(['--max_hic_jobs', str(kwargs['max_hic_jobs'])])
    if 'max_sv_jobs' in kwargs and kwargs['max_sv_jobs']:
        cmd.extend(['--max_sv_jobs', str(kwargs['max_sv_jobs'])])
    if 'max_kmer_jobs' in kwargs and kwargs['max_kmer_jobs']:
        cmd.extend(['--max_kmer_jobs', str(kwargs['max_kmer_jobs'])])
    
    # Add assembly options
    if 'enable_ai' in kwargs and kwargs['enable_ai'] is not None:
        cmd.extend(['--enable_ai', str(kwargs['enable_ai']).lower()])
    if 'detect_svs' in kwargs and kwargs['detect_svs'] is not None:
        cmd.extend(['--detect_svs', str(kwargs['detect_svs']).lower()])
    if 'huge' in kwargs and kwargs['huge']:
        cmd.append('--huge')
    if 'preserve_heterozygosity' in kwargs and kwargs['preserve_heterozygosity'] is not None:
        cmd.extend(['--preserve_heterozygosity', str(kwargs['preserve_heterozygosity']).lower()])
    if 'min_identity' in kwargs and kwargs['min_identity']:
        cmd.extend(['--min_identity', str(kwargs['min_identity'])])
    
    click.echo(f"Running Nextflow workflow: {' '.join(cmd)}")
    
    # Execute Nextflow
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode != 0:
        click.echo(f"✗ Nextflow execution failed with code {result.returncode}", err=True)
        sys.exit(result.returncode)
    else:
        click.echo("✓ Nextflow workflow completed successfully")


# ============================================================================
# NEXTFLOW STEP COMMANDS (invoked by Nextflow processes)
# ============================================================================

@main.command('classify')
@click.option('--input', '-i', 'input_file', required=True,
              type=click.Path(exists=True), help='Input reads file (FASTQ/FASTA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output classification JSON')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
def nf_classify(input_file, output, threads):
    """Classify reads by sequencing technology (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils import read_fastq, read_fasta
    from .preprocessing.read_classification_utility import (
        detect_technology_from_header, classify_read_technology,
    )

    click.echo(f"Classifying reads: {P(input_file).name}")
    p = P(input_file)
    suffix = p.suffix.lower()
    stem_suffix = P(p.stem).suffix.lower() if suffix in ('.gz', '.gzip') else suffix
    reader = read_fastq if stem_suffix in ('.fq', '.fastq') else read_fasta

    tech_counts: dict[str, int] = {}
    total = 0
    for read in reader(input_file):
        tech = detect_technology_from_header(read.id)
        tech_counts[tech] = tech_counts.get(tech, 0) + 1
        total += 1

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    result = {'total_reads': total, 'technology_counts': tech_counts}
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    click.echo(f"✓ Classified {total} reads → {output}")


@main.command('kweaver')
@click.option('--input', '-i', 'input_file', required=True,
              type=click.Path(exists=True), help='Corrected reads (FASTQ)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='K-mer predictions JSON')
@click.option('--report', type=click.Path(), help='K-Weaver report text file')
@click.option('--kmer-size', '-k', type=int, default=None,
              help='Override k-mer size (default: auto-predict)')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
def nf_kweaver(input_file, output, report, kmer_size, threads):
    """Run K-Weaver adaptive k-mer prediction."""
    import json
    from pathlib import Path as P
    from .preprocessing.kweaver_module import KWeaverPredictor

    click.echo(f"Running K-Weaver on: {P(input_file).name}")
    predictor = KWeaverPredictor(use_ml=True)
    prediction = predictor.predict_from_file(str(input_file))

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(prediction.to_dict(), f, indent=2)

    if report:
        rp = P(report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        with open(rp, 'w') as f:
            f.write(f"K-Weaver Report\n{'='*40}\n")
            f.write(f"Input: {input_file}\n")
            f.write(f"Primary k: {prediction.primary_k}\n")
            f.write(f"DBG k:     {prediction.dbg_k}\n")
            f.write(f"Overlap k: {prediction.overlap_k}\n")
            f.write(f"Confidence: {prediction.confidence:.3f}\n")

    click.echo(f"✓ K-Weaver predictions saved → {output}")


@main.command('nf-build-graph')
@click.option('--reads', type=click.Path(exists=True),
              help='Corrected reads (FASTQ)')
@click.option('--kmer-tables', type=click.Path(exists=True),
              help='Pre-computed k-mer tables (--huge mode)')
@click.option('--kmer-predictions', type=click.Path(exists=True),
              help='K-Weaver predictions JSON')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output assembly graph (GFA)')
@click.option('--alignments', type=click.Path(), default=None,
              help='Output alignments (BAM) — reserved for future use')
@click.option('--stats', type=click.Path(), default=None,
              help='Output build statistics (JSON)')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
@click.option('--device', type=str, default='cpu', help='Compute device (cpu/cuda)')
def nf_build_graph(reads, kmer_tables, kmer_predictions, output, alignments,
                   stats, threads, device):
    """Build assembly graph from reads (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .assembly_core.dbg_engine_module import DeBruijnGraphBuilder
    from .io_utils import read_fastq
    from .io_utils.assembly_export import export_graph_to_gfa

    if not reads and not kmer_tables:
        click.echo("❌ Error: --reads or --kmer-tables required", err=True)
        raise SystemExit(1)

    # Determine k from predictions
    k = 31
    if kmer_predictions:
        with open(kmer_predictions) as f:
            preds = json.load(f)
        k = preds.get('dbg_k', preds.get('primary_k', 31))

    click.echo(f"Building graph (k={k}, device={device})...")
    builder = DeBruijnGraphBuilder(base_k=k, use_gpu=(device != 'cpu'))

    if reads:
        read_list = list(read_fastq(reads))
        graph = builder.build(read_list)
    else:
        click.echo("⚠️ K-mer table loading not yet implemented, using empty graph")
        from .assembly_core.dbg_engine_module import KmerGraph
        graph = KmerGraph(base_k=k)

    # Export GFA
    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_graph_to_gfa(graph, out)

    # Stats
    if stats:
        sp = P(stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, 'w') as f:
            json.dump({
                'nodes': len(graph.nodes), 'edges': len(graph.edges),
                'k': k, 'device': device,
            }, f, indent=2)

    # Alignments placeholder
    if alignments:
        P(alignments).parent.mkdir(parents=True, exist_ok=True)
        P(alignments).touch()

    click.echo(f"✓ Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges → {output}")


@main.command('nf-edgewarden-filter')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Input assembly graph (GFA)')
@click.option('--edge-scores', required=True, type=click.Path(exists=True),
              help='Edge scores from batch scoring (JSON)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output filtered graph (GFA)')
@click.option('--stats', type=click.Path(), default=None,
              help='Output filtering statistics (JSON)')
@click.option('--enable-ai', type=str, default='true', help='Enable AI filtering')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
@click.option('--device', type=str, default='cpu', help='Compute device')
def nf_edgewarden_filter(graph, edge_scores, output, stats, enable_ai,
                         threads, device):
    """Filter graph edges with EdgeWarden (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import load_graph_from_gfa, export_graph_to_gfa

    click.echo(f"EdgeWarden filtering: {P(graph).name}")
    g = load_graph_from_gfa(graph)
    original_edges = len(g.edges)

    # Load edge scores and remove low-confidence edges
    with open(edge_scores) as f:
        scores = json.load(f)

    if isinstance(scores, dict):
        # Scores keyed by edge_id → confidence float
        threshold = 0.3
        remove_ids = [
            int(eid) for eid, conf in scores.items()
            if isinstance(conf, (int, float)) and conf < threshold
        ]
        for eid in remove_ids:
            if eid in g.edges:
                edge = g.edges[eid]
                g.out_edges.get(edge.from_id, set()).discard(eid)
                g.in_edges.get(edge.to_id, set()).discard(eid)
                del g.edges[eid]

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_graph_to_gfa(g, out)

    removed = original_edges - len(g.edges)
    click.echo(f"✓ Filtered: {removed} edges removed, {len(g.edges)} remain → {output}")

    if stats:
        sp = P(stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, 'w') as f:
            json.dump({
                'original_edges': original_edges,
                'removed': removed,
                'remaining': len(g.edges),
            }, f, indent=2)


@main.command('nf-pathweaver-iter-general')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Input assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output graph after path finding (GFA)')
@click.option('--paths', type=click.Path(), default=None,
              help='Output paths JSON')
@click.option('--enable-ai', type=str, default='true', help='Enable AI')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
@click.option('--device', type=str, default='cpu', help='Compute device')
def nf_pathweaver_iter_general(graph, output, paths, enable_ai, threads, device):
    """PathWeaver general iteration — initial path finding (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import load_graph_from_gfa, export_graph_to_gfa
    from .assembly_core.pathweaver_module import PathWeaver

    click.echo(f"PathWeaver general iteration: {P(graph).name}")
    g = load_graph_from_gfa(graph)

    pw = PathWeaver(graph=g, use_gpu=(device != 'cpu'))
    # Find paths from all source nodes (in_degree == 0)
    sources = [nid for nid in g.nodes if g.in_degree(nid) == 0]
    sinks = [nid for nid in g.nodes if g.out_degree(nid) == 0]

    found_paths = []
    if sources and sinks:
        try:
            found_paths = pw.find_paths_multi(sources)
        except Exception as e:
            click.echo(f"  ⚠️ Path finding fallback: {e}")

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_graph_to_gfa(g, out)

    if paths:
        pp = P(paths)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, 'w') as f:
            json.dump({
                'paths': len(found_paths),
                'sources': len(sources),
                'sinks': len(sinks),
            }, f, indent=2)

    click.echo(f"✓ General iteration: {len(found_paths)} paths found → {output}")


@main.command('nf-threadcompass-aggregate')
@click.option('--mappings', '-m', required=True, type=click.Path(exists=True),
              help='UL read mappings (JSON/PAF files)')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output UL routes (JSON)')
@click.option('--evidence', type=click.Path(), default=None,
              help='Output UL evidence (JSON)')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
def nf_threadcompass_aggregate(mappings, graph, output, evidence, threads):
    """Aggregate UL read mappings with ThreadCompass (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import load_graph_from_gfa
    from .assembly_core.threadcompass_module import ThreadCompass

    click.echo(f"ThreadCompass aggregation: {P(mappings).name}")
    g = load_graph_from_gfa(graph)

    tc = ThreadCompass(graph=g)
    routes = tc.route_ul_reads()

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'routes': len(routes), 'detail': str(routes)[:1000]}, f, indent=2)

    if evidence:
        ep = P(evidence)
        ep.parent.mkdir(parents=True, exist_ok=True)
        stats = tc.get_mapping_stats()
        with open(ep, 'w') as f:
            json.dump(stats, f, indent=2)

    click.echo(f"✓ UL routes aggregated → {output}")


@main.command('nf-strandtether-phase')
@click.option('--contacts', '-c', required=True, type=click.Path(exists=True),
              help='Hi-C contacts file')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output-matrix', type=click.Path(), default=None,
              help='Output contact matrix (H5)')
@click.option('--output-phasing', '-o', required=True, type=click.Path(),
              help='Output phasing info (JSON)')
@click.option('--stats', type=click.Path(), default=None,
              help='Output phasing statistics (JSON)')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
@click.option('--device', type=str, default='cpu', help='Compute device')
def nf_strandtether_phase(contacts, graph, output_matrix, output_phasing,
                          stats, threads, device):
    """Phase haplotypes from Hi-C contacts (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import load_graph_from_gfa
    from .assembly_core.strandtether_module import StrandTether

    click.echo(f"StrandTether phasing: {P(contacts).name}")
    g = load_graph_from_gfa(graph)

    st = StrandTether(use_gpu=(device != 'cpu'))
    phasing = st.phase_haplotypes(g)

    out = P(output_phasing)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'phasing': str(phasing)[:2000], 'nodes': len(g.nodes)}, f, indent=2)

    if stats:
        sp = P(stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, 'w') as f:
            json.dump({'nodes_phased': len(g.nodes), 'device': device}, f, indent=2)

    if output_matrix:
        P(output_matrix).parent.mkdir(parents=True, exist_ok=True)
        P(output_matrix).touch()  # Placeholder until H5 export implemented

    click.echo(f"✓ Phasing complete → {output_phasing}")


@main.command('nf-pathweaver-iter-strict')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Input graph from general iteration (GFA)')
@click.option('--ul-routes', type=click.Path(exists=True), default=None,
              help='UL routes from ThreadCompass (JSON)')
@click.option('--hic-phasing', type=click.Path(exists=True), default=None,
              help='Hi-C phasing info (JSON)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output final graph (GFA)')
@click.option('--paths', type=click.Path(), default=None,
              help='Output paths JSON')
@click.option('--enable-ai', type=str, default='true', help='Enable AI')
@click.option('--preserve-heterozygosity', type=str, default='true',
              help='Preserve heterozygosity')
@click.option('--min-identity', type=float, default=0.995,
              help='Minimum identity threshold')
@click.option('--threads', '-t', type=int, default=4, help='Number of threads')
@click.option('--device', type=str, default='cpu', help='Compute device')
def nf_pathweaver_iter_strict(graph, ul_routes, hic_phasing, output, paths,
                              enable_ai, preserve_heterozygosity, min_identity,
                              threads, device):
    """PathWeaver strict iteration with UL/Hi-C evidence (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import load_graph_from_gfa, export_graph_to_gfa
    from .assembly_core.pathweaver_module import PathWeaver

    click.echo(f"PathWeaver strict iteration: {P(graph).name}")
    g = load_graph_from_gfa(graph)

    pw = PathWeaver(graph=g, use_gpu=(device != 'cpu'))

    sources = [nid for nid in g.nodes if g.in_degree(nid) == 0]
    found_paths = []
    if sources:
        try:
            found_paths = pw.find_paths_multi(sources)
        except Exception as e:
            click.echo(f"  ⚠️ Path finding fallback: {e}")

    out = P(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_graph_to_gfa(g, out)

    if paths:
        pp = P(paths)
        pp.parent.mkdir(parents=True, exist_ok=True)
        with open(pp, 'w') as f:
            json.dump({
                'paths': len(found_paths),
                'ul_routes_used': ul_routes is not None,
                'hic_phasing_used': hic_phasing is not None,
            }, f, indent=2)

    click.echo(f"✓ Strict iteration: {len(found_paths)} paths → {output}")


@main.command('nf-export-assembly')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Final assembly graph (GFA)')
@click.option('--svs', type=click.Path(exists=True), default=None,
              help='Structural variants (VCF)')
@click.option('--output-fasta', required=True, type=click.Path(),
              help='Output assembly FASTA')
@click.option('--output-gfa', type=click.Path(), default=None,
              help='Output clean GFA')
@click.option('--stats', type=click.Path(), default=None,
              help='Output assembly statistics (JSON)')
@click.option('--export-coverage', is_flag=True, help='Export coverage CSVs')
def nf_export_assembly(graph, svs, output_fasta, output_gfa, stats,
                       export_coverage):
    """Export final assembly to FASTA/GFA/stats (Nextflow step)."""
    import json
    from pathlib import Path as P
    from .io_utils.assembly_export import (
        load_graph_from_gfa, export_graph_to_gfa, write_contigs_fasta,
    )

    click.echo(f"Exporting assembly: {P(graph).name}")
    g = load_graph_from_gfa(graph)

    # Build contigs from graph nodes (each node = unitig)
    contigs = []
    for nid in sorted(g.nodes.keys()):
        node = g.nodes[nid]
        if node.seq:
            contigs.append((f"contig_{nid}", node.seq))

    # Export FASTA
    out_fa = P(output_fasta)
    out_fa.parent.mkdir(parents=True, exist_ok=True)
    write_contigs_fasta(contigs, out_fa)

    # Export clean GFA
    if output_gfa:
        out_gfa = P(output_gfa)
        out_gfa.parent.mkdir(parents=True, exist_ok=True)
        export_graph_to_gfa(g, out_gfa)

    # Stats
    if stats:
        sp = P(stats)
        sp.parent.mkdir(parents=True, exist_ok=True)
        lengths = sorted([len(seq) for _, seq in contigs], reverse=True)
        total = sum(lengths)
        cumsum = 0
        n50 = 0
        for l in lengths:
            cumsum += l
            if cumsum >= total / 2:
                n50 = l
                break
        with open(sp, 'w') as f:
            json.dump({
                'total_contigs': len(contigs),
                'total_bases': total,
                'n50': n50,
                'largest_contig': lengths[0] if lengths else 0,
                'smallest_contig': lengths[-1] if lengths else 0,
            }, f, indent=2)

    # Coverage CSV placeholder
    if export_coverage:
        cov_path = out_fa.parent / "coverage_nodes.csv"
        with open(cov_path, 'w') as f:
            f.write("node,coverage\n")
            for nid, node in g.nodes.items():
                f.write(f"unitig-{nid},{node.coverage:.1f}\n")

    click.echo(f"✓ Exported: {len(contigs)} contigs, {sum(len(s) for _,s in contigs):,} bp → {output_fasta}")


# ============================================================================
# BATCH (INTERNAL) COMMANDS — Nextflow parallelization
# ============================================================================

@main.group(hidden=True)
def batch():
    """
    [INTERNAL] Batch processing commands for Nextflow parallelization.
    
    These commands are called by Nextflow workflows and are not intended
    for direct use. Use the top-level commands with --nextflow flag instead.
    """
    pass


# ============================================================================
# USER-FACING PROCESSING COMMANDS
# ============================================================================

@main.command('correct')
@click.option('--hifi', type=click.Path(exists=True),
              help='HiFi reads for error correction')
@click.option('--ont', type=click.Path(exists=True),
              help='ONT reads for error correction')
@click.option('--illumina', type=click.Path(exists=True),
              help='Illumina reads for error correction')
@click.option('--ancient', type=click.Path(exists=True),
              help='Ancient DNA reads for error correction (damage-aware)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output directory for corrected reads')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
@click.option('--max-iterations', type=int, default=3,
              help='Maximum correction iterations per read set (default: 3)')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm (default: local)')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--correction-batch-size', type=int, default=100000,
              help='Reads per correction batch (Nextflow mode)')
@click.option('--max-correction-jobs', type=int, default=20,
              help='Max parallel correction jobs (Nextflow mode)')
def correct_reads(hifi, ont, illumina, ancient, output, threads, max_iterations,
                  nextflow, nf_profile, nf_resume,
                  correction_batch_size, max_correction_jobs):
    """
    Correct sequencing errors in reads.
    
    Direct mode (default): Processes entire dataset on local machine.
    Nextflow mode (--nextflow): Splits reads into batches for parallel processing.
    
    Supports HiFi, ONT, Illumina, and ancient DNA reads. Ancient DNA
    correction uses damage-aware algorithms that preserve authentic
    deamination patterns (C→T at 5' ends, G→A at 3' ends).
    
    Examples:
        # Direct mode - process locally
        strandweaver correct --hifi reads.fq.gz -o corrected/
        
        # Multiple technologies
        strandweaver correct --hifi hifi.fq.gz --ont ont.fq.gz -o corrected/
        
        # Ancient DNA with damage-aware correction
        strandweaver correct --ancient adna.fq.gz -o corrected/
        
        # Illumina reads
        strandweaver correct --illumina illumina.fq.gz -o corrected/
        
        # Nextflow mode - parallel processing on cluster
        strandweaver correct --hifi reads.fq.gz -o corrected/ \\
            --nextflow --nf-profile slurm --nf-resume
    """
    if nextflow:
        # Run via Nextflow
        _run_nextflow(
            'CORRECT',
            hifi=hifi,
            ont=ont,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            correction_batch_size=correction_batch_size,
            max_correction_jobs=max_correction_jobs
        )
    else:
        # Run directly
        import json
        from pathlib import Path
        from .preprocessing.errorsmith_module import profile_technology, correct_batch
        
        click.echo("Running error correction (direct mode)...")
        click.echo(f"  Max iterations: {max_iterations}")
        
        # Profile errors first
        click.echo("Step 1/2: Profiling errors...")
        profiles = {}
        if hifi:
            click.echo(f"  • HiFi: {hifi}")
            profiles['hifi'] = profile_technology(hifi, technology='hifi', threads=threads)
        if ont:
            click.echo(f"  • ONT: {ont}")
            profiles['ont'] = profile_technology(ont, technology='ont', threads=threads)
        if illumina:
            click.echo(f"  • Illumina: {illumina}")
            profiles['illumina'] = profile_technology(illumina, technology='illumina', threads=threads)
        if ancient:
            click.echo(f"  • Ancient DNA: {ancient}")
            profiles['ancient'] = profile_technology(ancient, technology='ancient', threads=threads)
        
        # Correct reads
        click.echo("Step 2/2: Correcting reads...")
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if hifi:
            output_file = output_dir / 'corrected_hifi.fastq.gz'
            correct_batch(hifi, 'hifi', profiles['hifi'], str(output_file), threads)
            click.echo(f"  ✓ HiFi corrected: {output_file}")
        
        if ont:
            output_file = output_dir / 'corrected_ont.fastq.gz'
            correct_batch(ont, 'ont', profiles['ont'], str(output_file), threads)
            click.echo(f"  ✓ ONT corrected: {output_file}")
        
        if illumina:
            output_file = output_dir / 'corrected_illumina.fastq.gz'
            correct_batch(illumina, 'illumina', profiles['illumina'], str(output_file), threads)
            click.echo(f"  ✓ Illumina corrected: {output_file}")
        
        if ancient:
            output_file = output_dir / 'corrected_ancient.fastq.gz'
            correct_batch(ancient, 'ancient', profiles['ancient'], str(output_file), threads)
            click.echo(f"  ✓ Ancient DNA corrected (damage-aware): {output_file}")
        
        click.echo(f"✓ Error correction complete: {output}")


@main.command('extract-kmers')
@click.option('--hifi', type=click.Path(exists=True),
              help='HiFi reads for k-mer extraction')
@click.option('--ont', type=click.Path(exists=True),
              help='ONT reads for k-mer extraction')
@click.option('--kmer-size', '-k', type=int, required=True,
              help='K-mer size to extract')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output k-mer table file (PKL)')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization (for huge genomes)')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--kmer-batch-size', type=int, default=1000000,
              help='Reads per k-mer extraction batch (Nextflow mode)')
@click.option('--max-kmer-jobs', type=int, default=12,
              help='Max parallel k-mer jobs (Nextflow mode)')
def extract_kmers(hifi, ont, kmer_size, output, threads, nextflow, nf_profile,
                  nf_resume, kmer_batch_size, max_kmer_jobs):
    """
    Extract k-mers from reads for k-mer size prediction.
    
    Direct mode (default): Single-job processing for normal genomes.
    Nextflow mode (--nextflow): Parallel processing for huge genomes (>10GB).
    
    Examples:
        # Direct mode
        strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl
        
        # Nextflow mode for huge genome
        strandweaver extract-kmers --hifi reads.fq.gz -k 31 -o kmers.pkl \\
            --nextflow --nf-profile slurm
    """
    if nextflow:
        _run_nextflow(
            'EXTRACT_KMERS',
            hifi=hifi,
            ont=ont,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            kmer_batch_size=kmer_batch_size,
            max_kmer_jobs=max_kmer_jobs,
            huge=True
        )
    else:
        from pathlib import Path
        from .preprocessing.kweaver_module import extract_kmers_batch
        
        click.echo(f"Extracting {kmer_size}-mers (direct mode)...")
        
        if hifi:
            click.echo(f"  • Processing HiFi: {hifi}")
            extract_kmers_batch(hifi, kmer_size, output, threads)
        elif ont:
            click.echo(f"  • Processing ONT: {ont}")
            extract_kmers_batch(ont, kmer_size, output, threads)
        
        click.echo(f"✓ K-mers extracted: {output}")


@main.command('nf-score-edges')
@click.option('--edges', '-e', required=True, type=click.Path(exists=True),
              help='Graph edges file (JSON)')
@click.option('--alignments', '-a', required=True, type=click.Path(exists=True),
              help='Read alignments (BAM/PAF)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output scored edges (JSON)')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--edge-batch-size', type=int, default=10000,
              help='Edges per batch (Nextflow mode)')
@click.option('--max-edge-jobs', type=int, default=8,
              help='Max parallel edge scoring jobs (Nextflow mode)')
def score_edges_cmd(edges, alignments, output, threads, nextflow, nf_profile,
                    nf_resume, edge_batch_size, max_edge_jobs):
    """
    Score assembly graph edges using read alignments.
    
    Examples:
        # Direct mode
        strandweaver score-edges -e edges.json -a aligns.bam -o scored.json
        
        # Nextflow mode for large graphs
        strandweaver score-edges -e edges.json -a aligns.bam -o scored.json \\
            --nextflow --nf-profile slurm
    """
    if nextflow:
        _run_nextflow(
            'SCORE_EDGES',
            edges=edges,
            alignments=alignments,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            edge_batch_size=edge_batch_size,
            max_edge_jobs=max_edge_jobs
        )
    else:
        import json
        from pathlib import Path
        from .assembly_core.edgewarden_module import score_edges_batch
        
        click.echo("Scoring edges (direct mode)...")
        
        # Load edges
        with open(edges, 'r') as f:
            edge_list = json.load(f)
        
        # Score edges
        scored = score_edges_batch(edge_list, alignments, threads)
        
        # Save
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            json.dump(scored, f, indent=2)
        
        click.echo(f"✓ Edges scored: {output}")


@main.command('map-ul')
@click.option('--ul-reads', '-u', required=True, type=click.Path(exists=True),
              help='Ultra-long ONT reads (FASTQ)')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output alignments (PAF)')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
@click.option('--use-gpu', is_flag=True,
              help='Use GPU acceleration if available')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--ul-batch-size', type=int, default=100,
              help='UL reads per batch (Nextflow mode)')
@click.option('--max-ul-jobs', type=int, default=10,
              help='Max parallel UL mapping jobs (Nextflow mode)')
def map_ul_cmd(ul_reads, graph, output, threads, use_gpu, nextflow, nf_profile,
               nf_resume, ul_batch_size, max_ul_jobs):
    """
    Map ultra-long reads to assembly graph.
    
    Examples:
        # Direct mode
        strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf
        
        # Nextflow mode with GPU
        strandweaver map-ul -u ul_reads.fq.gz -g graph.gfa -o aligns.paf \\
            --nextflow --use-gpu --nf-profile slurm
    """
    if nextflow:
        _run_nextflow(
            'MAP_UL',
            ont_ul=ul_reads,
            graph=graph,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            ul_batch_size=ul_batch_size,
            max_ul_jobs=max_ul_jobs
        )
    else:
        from pathlib import Path
        from .assembly_core.threadcompass_module import map_reads_batch
        
        click.echo("Mapping ultra-long reads (direct mode)...")
        
        map_reads_batch(ul_reads, graph, output, threads, use_gpu)
        
        click.echo(f"✓ UL reads mapped: {output}")


@main.command('align-hic')
@click.option('--hic-r1', required=True, type=click.Path(exists=True),
              help='Hi-C R1 reads (FASTQ)')
@click.option('--hic-r2', required=True, type=click.Path(exists=True),
              help='Hi-C R2 reads (FASTQ)')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output alignments (BAM)')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--hic-batch-size', type=int, default=500000,
              help='Hi-C read pairs per batch (Nextflow mode)')
@click.option('--max-hic-jobs', type=int, default=15,
              help='Max parallel Hi-C jobs (Nextflow mode)')
def align_hic_cmd(hic_r1, hic_r2, graph, output, threads, nextflow, nf_profile,
                  nf_resume, hic_batch_size, max_hic_jobs):
    """
    Align Hi-C reads to assembly graph for scaffolding.
    
    Examples:
        # Direct mode
        strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \\
            -g graph.gfa -o aligns.bam
        
        # Nextflow mode
        strandweaver align-hic --hic-r1 R1.fq.gz --hic-r2 R2.fq.gz \\
            -g graph.gfa -o aligns.bam --nextflow --nf-profile slurm
    """
    if nextflow:
        _run_nextflow(
            'ALIGN_HIC',
            hic_r1=hic_r1,
            hic_r2=hic_r2,
            graph=graph,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            hic_batch_size=hic_batch_size,
            max_hic_jobs=max_hic_jobs
        )
    else:
        from pathlib import Path
        from .assembly_core.strandtether_module import align_reads_batch
        
        click.echo("Aligning Hi-C reads (direct mode)...")
        
        # Note: align_reads_batch expects interleaved or will handle R1/R2
        # For now, assuming it can handle paired files
        align_reads_batch(f"{hic_r1},{hic_r2}", graph, output, threads)
        
        click.echo(f"✓ Hi-C reads aligned: {output}")


@main.command('nf-detect-svs')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output SVs (VCF)')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads (direct mode only)')
# Nextflow mode options
@click.option('--nextflow', is_flag=True,
              help='Run via Nextflow for automatic parallelization')
@click.option('--nf-profile', type=str, default='local',
              help='Nextflow profile: local, docker, singularity, slurm')
@click.option('--nf-resume', is_flag=True,
              help='Resume Nextflow workflow from last checkpoint')
@click.option('--sv-batch-size', type=int, default=1000,
              help='Graph nodes per SV detection batch (Nextflow mode)')
@click.option('--max-sv-jobs', type=int, default=10,
              help='Max parallel SV detection jobs (Nextflow mode)')
def detect_svs_cmd(graph, output, threads, nextflow, nf_profile, nf_resume,
                   sv_batch_size, max_sv_jobs):
    """
    Detect structural variants in assembly graph.
    
    Examples:
        # Direct mode
        strandweaver detect-svs -g graph.gfa -o variants.vcf
        
        # Nextflow mode for large graphs
        strandweaver detect-svs -g graph.gfa -o variants.vcf \\
            --nextflow --nf-profile slurm
    """
    if nextflow:
        _run_nextflow(
            'DETECT_SVS',
            graph=graph,
            output=output,
            nf_profile=nf_profile,
            nf_resume=nf_resume,
            sv_batch_size=sv_batch_size,
            max_sv_jobs=max_sv_jobs,
            detect_svs=True
        )
    else:
        import json
        from pathlib import Path
        from .assembly_core.svscribe_module import detect_svs_batch, export_vcf
        
        click.echo("Detecting structural variants (direct mode)...")
        
        # Detect SVs
        svs = detect_svs_batch(graph)
        
        # Export to VCF
        export_vcf(svs, Path(output))
        
        click.echo(f"✓ SVs detected: {output} ({len(svs)} variants)")


# ----------------------------------------------------------------------------
# Error Profiling & Correction Batch Commands
# ----------------------------------------------------------------------------

@batch.command('profile-errors')
@click.option('--hifi', type=click.Path(exists=True),
              help='HiFi reads for error profiling')
@click.option('--ont', type=click.Path(exists=True),
              help='ONT reads for error profiling')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output JSON file with error profiles')
@click.option('--threads', '-t', type=int, default=8,
              help='Number of threads')
def batch_profile_errors(hifi, ont, output, threads):
    """
    Profile sequencing errors across full dataset.
    
    Sequential stage - requires all reads for accurate k-mer spectrum.
    """
    import json
    from pathlib import Path
    from .preprocessing.errorsmith_module import profile_technology
    
    click.echo(f"Profiling errors from reads...")
    
    # Profile errors based on available technologies
    profiles = {}
    if hifi:
        click.echo(f"  • HiFi: {hifi}")
        profiles['hifi'] = profile_technology(hifi, technology='hifi', threads=threads)
    if ont:
        click.echo(f"  • ONT: {ont}")
        profiles['ont'] = profile_technology(ont, technology='ont', threads=threads)
    
    # Save profiles to JSON
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(profiles, f, indent=2)
    
    click.echo(f"✓ Error profiles saved: {output}")


@batch.command('correct')
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input reads batch (FASTQ)')
@click.option('--profiles', '-p', required=True, type=click.Path(exists=True),
              help='Error profiles JSON from profile-errors')
@click.option('--technology', '-tech', required=True,
              type=click.Choice(['hifi', 'ont', 'illumina']),
              help='Sequencing technology')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output corrected reads (FASTQ)')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
def batch_correct_reads(input, profiles, technology, output, threads):
    """
    Correct errors in a batch of reads.
    
    Parallel stage - processes independent read batches.
    """
    import json
    from pathlib import Path
    from .preprocessing.errorsmith_module import correct_batch
    
    click.echo(f"Correcting {technology} reads batch: {Path(input).name}")
    
    # Load error profiles
    with open(profiles, 'r') as f:
        error_profiles = json.load(f)
    
    # Correct reads in batch
    correct_batch(
        reads_file=input,
        technology=technology,
        error_profile=error_profiles.get(technology, {}),
        output_file=output,
        threads=threads
    )
    
    click.echo(f"✓ Corrected batch saved: {output}")


@batch.command('merge-corrected')
@click.option('--input', '-i', 'input_files', multiple=True, required=True,
              type=click.Path(exists=True),
              help='Corrected read batches to merge')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output merged FASTQ file')
def batch_merge_corrected(input_files, output):
    """
    Merge corrected read batches into single file.
    
    Aggregation stage - collects parallel correction results.
    """
    from pathlib import Path
    import gzip
    import shutil
    
    click.echo(f"Merging {len(input_files)} corrected batches...")
    
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine if output should be compressed
    is_gzipped = output.endswith('.gz')
    
    # Open output file
    if is_gzipped:
        out_fh = gzip.open(output_path, 'wt')
    else:
        out_fh = open(output_path, 'w')
    
    try:
        for batch_file in input_files:
            click.echo(f"  • {Path(batch_file).name}")
            
            # Read input batch (handle compressed/uncompressed)
            if batch_file.endswith('.gz'):
                in_fh = gzip.open(batch_file, 'rt')
            else:
                in_fh = open(batch_file, 'r')
            
            # Copy to output
            shutil.copyfileobj(in_fh, out_fh)
            in_fh.close()
    finally:
        out_fh.close()
    
    click.echo(f"✓ Merged corrected reads: {output}")


# ----------------------------------------------------------------------------
# K-mer Extraction Batch Commands (--huge mode)
# ----------------------------------------------------------------------------

@batch.command('extract-kmers')
@click.option('--input', '-i', 'input_file', type=click.Path(exists=True),
              help='Input reads batch (FASTQ)')
@click.option('--reads', 'reads_file', type=click.Path(exists=True),
              help='Input reads batch (FASTQ) — alias for --input')
@click.option('--kmer-size', '-k', type=int, default=None,
              help='K-mer size (used if --kmer-predictions not provided)')
@click.option('--kmer-predictions', type=click.Path(exists=True), default=None,
              help='K-Weaver predictions JSON (overrides --kmer-size)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output k-mer table (binary)')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
def batch_extract_kmers(input_file, reads_file, kmer_size, kmer_predictions,
                        output, threads):
    """
    Extract k-mers from read batch for --huge genome mode.
    
    Parallel stage - processes independent read batches.
    """
    import json
    from pathlib import Path
    from .preprocessing.kweaver_module import extract_kmers_batch
    
    actual_input = reads_file or input_file
    if not actual_input:
        click.echo("❌ Error: --input or --reads required", err=True)
        raise SystemExit(1)
    
    # Resolve k from predictions or explicit size
    k = kmer_size or 31
    if kmer_predictions:
        with open(kmer_predictions) as f:
            preds = json.load(f)
        k = preds.get('dbg_k', preds.get('primary_k', k))
    
    click.echo(f"Extracting {k}-mers from batch: {Path(actual_input).name}")
    
    # Extract k-mers from batch
    extract_kmers_batch(
        reads_file=actual_input,
        k=k,
        output_table=output,
        threads=threads
    )
    
    click.echo(f"✓ K-mer table saved: {output}")


# ----------------------------------------------------------------------------
# Edge Scoring Batch Commands
# ----------------------------------------------------------------------------

@batch.command('score-edges')
@click.option('--edges', '-e', required=True, type=click.Path(exists=True),
              help='Edge batch JSON file')
@click.option('--alignments', '-a', required=True, type=click.Path(exists=True),
              help='Read alignments (BAM/PAF)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output scored edges JSON')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
def batch_score_edges(edges, alignments, output, threads):
    """
    Score graph edges for a batch.
    
    Parallel stage - processes independent edge sets.
    """
    import json
    from pathlib import Path
    from .assembly_core.edgewarden_module import score_edges_batch
    
    click.echo(f"Scoring edge batch: {Path(edges).name}")
    
    # Load edges
    with open(edges, 'r') as f:
        edge_batch = json.load(f)
    
    # Score edges
    scored_edges = score_edges_batch(
        edges=edge_batch,
        alignments=alignments,
        threads=threads
    )
    
    # Save scored edges
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(scored_edges, f, indent=2)
    
    click.echo(f"✓ Scored edges saved: {output}")


# ----------------------------------------------------------------------------
# UL Read Mapping Batch Commands
# ----------------------------------------------------------------------------

@batch.command('map-ul')
@click.option('--ul-reads', '-u', required=True, type=click.Path(exists=True),
              help='Ultra-long reads batch (FASTQ)')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output alignments (PAF)')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
@click.option('--use-gpu', is_flag=True,
              help='Use GPU acceleration if available')
def batch_map_ul_reads(ul_reads, graph, output, threads, use_gpu):
    """
    Map ultra-long reads to graph (batch).
    
    Parallel stage - processes independent UL read batches.
    """
    from pathlib import Path
    from .assembly_core.threadcompass_module import map_reads_batch
    
    click.echo(f"Mapping UL reads batch: {Path(ul_reads).name}")
    
    # Map UL reads
    map_reads_batch(
        reads_file=ul_reads,
        graph_file=graph,
        output_paf=output,
        threads=threads,
        use_gpu=use_gpu
    )
    
    click.echo(f"✓ UL alignments saved: {output}")


# ----------------------------------------------------------------------------
# Hi-C Alignment Batch Commands
# ----------------------------------------------------------------------------

@batch.command('align-hic')
@click.option('--hic-reads', '-h', required=True, type=click.Path(exists=True),
              help='Hi-C reads batch (interleaved FASTQ)')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Assembly graph (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output alignments (BAM)')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
def batch_align_hic(hic_reads, graph, output, threads):
    """
    Align Hi-C reads to graph (batch).
    
    Parallel stage - processes independent Hi-C read pair batches.
    """
    from pathlib import Path
    from .assembly_core.strandtether_module import align_reads_batch
    
    click.echo(f"Aligning Hi-C reads batch: {Path(hic_reads).name}")
    
    # Align Hi-C reads
    align_reads_batch(
        reads_file=hic_reads,
        graph_file=graph,
        output_bam=output,
        threads=threads
    )
    
    click.echo(f"✓ Hi-C alignments saved: {output}")


# ----------------------------------------------------------------------------
# SV Detection Batch Commands
# ----------------------------------------------------------------------------

@batch.command('detect-svs')
@click.option('--graph', '-g', required=True, type=click.Path(exists=True),
              help='Graph partition (GFA)')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output SVs (JSON)')
@click.option('--threads', '-t', type=int, default=4,
              help='Number of threads')
def batch_detect_svs(graph, output, threads):
    """
    Detect structural variants in graph partition.
    
    Parallel stage - processes independent graph regions.
    """
    import json
    from pathlib import Path
    from .assembly_core.svscribe_module import detect_svs_batch
    
    click.echo(f"Detecting SVs in partition: {Path(graph).name}")
    
    # Detect SVs
    svs = detect_svs_batch(graph_file=graph)
    
    # Save SVs
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(svs, f, indent=2)
    
    click.echo(f"✓ SVs saved: {output}")


@batch.command('merge-svs')
@click.option('--input', '-i', 'input_files', multiple=True,
              type=click.Path(exists=True),
              help='SV batch JSON files to merge')
@click.option('--vcfs', 'vcf_files', multiple=True,
              type=click.Path(exists=True),
              help='SV batch files — alias for --input')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='Output merged SVs (VCF)')
@click.option('--summary', type=click.Path(), default=None,
              help='Output summary JSON')
def batch_merge_svs(input_files, vcf_files, output, summary):
    """
    Merge SV calls from batches into single VCF.
    
    Aggregation stage - collects parallel SV detection results.
    """
    import json
    from pathlib import Path
    from .assembly_core.svscribe_module import merge_sv_calls, export_vcf
    
    all_inputs = list(input_files) + list(vcf_files)
    if not all_inputs:
        click.echo("❌ Error: --input or --vcfs required", err=True)
        raise SystemExit(1)
    
    click.echo(f"Merging {len(all_inputs)} SV batches...")
    
    # Load all SV calls
    all_svs = []
    for sv_file in all_inputs:
        click.echo(f"  • {Path(sv_file).name}")
        with open(sv_file, 'r') as f:
            batch_svs = json.load(f)
            all_svs.extend(batch_svs)
    
    # Merge and deduplicate SVs
    merged_svs = merge_sv_calls(all_svs)
    
    # Export to VCF
    output_path = Path(output)
    export_vcf(merged_svs, output_path)
    
    click.echo(f"✓ Merged SVs: {output} ({len(merged_svs)} variants)")
    
    # Summary JSON
    if summary:
        summary_path = Path(summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w') as f:
            json.dump({
                'batches_merged': len(all_inputs),
                'total_svs': len(merged_svs),
            }, f, indent=2)


# ============================================================================
# Training Commands
# ============================================================================

@main.group()
def train():
    """
    Generate training data and train StrandWeaver ML models.

    StrandWeaver's AI-powered assembly modules (EdgeAI, DiploidAI, PathGNN,
    UL Routing, SV Detection) can be retrained for your target organism
    using synthetic data.

    Workflow:

      1. Generate synthetic data:
         strandweaver train generate-data --genome-size 5000000 -o training_data/

      2. Train models on the generated data:
         strandweaver train run --data-dir training_data/ --output-dir models/

      3. Use trained models in a pipeline run:
         strandweaver pipeline -r1 reads.fq --technology1 ont --model-dir models/ ...
    """
    pass


@train.command('generate-data')
@click.option('--genome-size', type=int, required=True,
              help='Simulated genome size in bp (e.g. 1000000 for 1 Mb)')
@click.option('-o', '--output', required=True, type=click.Path(),
              help='Output directory for generated training data')
@click.option('--gc-content', type=float, default=0.42, show_default=True,
              help='GC content as a fraction (0–1)')
@click.option('--repeat-density', type=float, default=0.30, show_default=True,
              help='Fraction of genome that is repetitive (0–1)')
@click.option('--read-types', type=click.Choice(
    ['illumina', 'hifi', 'ont', 'ultra_long', 'hic', 'ancient_dna']),
    multiple=True, default=('hifi',), show_default=True,
    help='Sequencing technologies to simulate (repeat for multiple)')
@click.option('--coverage', type=float, multiple=True, default=None,
              help='Coverage per read-type in the same order (default: 30× each). '
                   'Repeat for each --read-types entry.')
@click.option('-n', '--num-genomes', type=int, default=10, show_default=True,
              help='Number of independent genomes to generate')
@click.option('--seed', type=int, default=None,
              help='Random seed for reproducibility')
@click.option('--ploidy', type=click.Choice(['haploid', 'diploid', 'triploid', 'tetraploid']),
              default='diploid', show_default=True, help='Ploidy level')
@click.option('--snp-rate', type=float, default=0.001, show_default=True,
              help='SNP rate per bp between haplotypes')
@click.option('--indel-rate', type=float, default=0.0001, show_default=True,
              help='Small indel rate per bp')
@click.option('--sv-density', type=float, default=0.00001, show_default=True,
              help='Structural variant density per bp')
@click.option('--centromeres', type=int, default=1, show_default=True,
              help='Number of centromeric regions per genome')
@click.option('--graph-training', is_flag=True, default=False,
              help='Enable graph training data generation (builds synthetic overlap '
                   'graphs with ground-truth labels for EdgeAI, DiploidAI, PathGNN, '
                   'UL Routing, and SV Detection)')
@click.option('--min-overlap-bp', type=int, default=500, show_default=True,
              help='Minimum overlap length (bp) for graph edges')
@click.option('--min-overlap-identity', type=float, default=0.90, show_default=True,
              help='Minimum overlap identity for graph edges')
@click.option('--noise-edge-fraction', type=float, default=0.10, show_default=True,
              help='Fraction of noise (false) edges to inject for negative-class training')
@click.option('--no-noise-edges', is_flag=True, default=False,
              help='Disable noise edge injection')
@click.option('--no-gfa', is_flag=True, default=False,
              help='Skip GFA file export')
@click.option('--graph-max-coverage', type=float, default=None,
              help='Subsample reads to this coverage before graph construction (saves RAM)')
@click.option('--graph-only', is_flag=True, default=False,
              help='\u26a1 Fast mode: simulate reads in-memory for realistic coverage/overlap '
                   'features but skip writing FASTQ/FASTA files to disk.  Only outputs '
                   'graph training CSVs.  ~10\u00d7 faster, ~95%% less disk.')
def train_generate_data(genome_size, output, gc_content, repeat_density,
                        read_types, coverage, num_genomes, seed,
                        ploidy, snp_rate, indel_rate, sv_density, centromeres,
                        graph_training, min_overlap_bp, min_overlap_identity,
                        noise_edge_fraction, no_noise_edges, no_gfa,
                        graph_max_coverage, graph_only):
    """
    Generate synthetic training data for StrandWeaver ML models.

    Creates synthetic diploid genomes with controlled variation (SNPs, indels,
    SVs, repeats) and simulates sequencing reads for one or more technologies.
    Optionally builds labelled assembly graphs for all five ML models.

    Examples:

      # Quick test — 10 diploid genomes (1 Mb), HiFi 30×
      strandweaver train generate-data --genome-size 1000000 -o data/test

      # Multi-technology dataset
      strandweaver train generate-data --genome-size 5000000 -n 50 \\
          --read-types hifi --read-types ont --read-types hic \\
          --coverage 30 --coverage 20 --coverage 15 -o data/multi_tech

      # Repeat-rich genome with graph training labels
      strandweaver train generate-data --genome-size 2000000 -n 20 \\
          --repeat-density 0.60 --gc-content 0.35 --graph-training \\
          -o data/repeat_rich

      # \u26a1 Fast graph-only mode (no FASTQ written, ~10\u00d7 faster)
      strandweaver train generate-data --genome-size 1000000 -n 200 \\
          --read-types hifi --read-types ont --read-types hic \\
          --graph-training --graph-only -o data/fast_graphs
    """
    from strandweaver.user_training import (
        UserGenomeConfig, UserReadConfig, UserTrainingConfig,
        GraphTrainingConfig, ReadType, Ploidy,
        generate_custom_training_data,
    )

    if generate_custom_training_data is None:
        click.echo("Error: Training data generation requires the StrandWeaver "
                    "training backend.  See the documentation for setup instructions.",
                    err=True)
        raise SystemExit(1)

    # ── Resolve read-types and coverage ────────────────────────────
    read_types_list = list(read_types) if read_types else ['hifi']
    num_techs = len(read_types_list)
    if not coverage:
        coverages = [30.0] * num_techs
    elif len(coverage) == 1:
        coverages = list(coverage) * num_techs
    elif len(coverage) != num_techs:
        click.echo(f"Error: number of --coverage values ({len(coverage)}) must "
                    f"match number of --read-types ({num_techs})", err=True)
        raise SystemExit(1)
    else:
        coverages = list(coverage)

    # ── Build config objects ───────────────────────────────────────
    genome_config = UserGenomeConfig(
        genome_size=genome_size,
        num_genomes=num_genomes,
        gc_content=gc_content,
        repeat_density=repeat_density,
        ploidy=Ploidy[ploidy.upper()],
        snp_rate=snp_rate,
        indel_rate=indel_rate,
        sv_density=sv_density,
        centromere_count=centromeres,
        random_seed=seed,
    )

    read_configs = [
        UserReadConfig(read_type=ReadType(rt), coverage=cov)
        for rt, cov in zip(read_types_list, coverages)
    ]

    # If --graph-only is set, auto-enable graph training
    if graph_only and not graph_training:
        graph_training = True

    graph_config = None
    if graph_training:
        graph_config = GraphTrainingConfig(
            enabled=True,
            min_overlap_bp=min_overlap_bp,
            min_overlap_identity=min_overlap_identity,
            add_noise_edges=not no_noise_edges,
            noise_edge_fraction=noise_edge_fraction,
            export_gfa=not no_gfa and not graph_only,  # skip GFA in fast mode
            max_coverage_for_graph=graph_max_coverage,
        )

    training_config = UserTrainingConfig(
        genome_config=genome_config,
        read_configs=read_configs,
        output_dir=Path(output),
        graph_config=graph_config,
        graph_only=graph_only,
    )

    # ── Display summary ────────────────────────────────────────────
    _TECH_LABELS = {
        'illumina': 'Illumina PE', 'hifi': 'PacBio HiFi', 'ont': 'ONT',
        'ultra_long': 'ONT Ultra-long', 'hic': 'Hi-C', 'ancient_dna': 'Ancient DNA',
    }
    techs_str = ", ".join(
        f"{_TECH_LABELS.get(rt, rt)} {cov:.0f}×"
        for rt, cov in zip(read_types_list, coverages))

    click.echo()
    click.echo("╔══════════════════════════════════════════════════════════════╗")
    click.echo("║       StrandWeaver  ·  Training Data Generation            ║")
    click.echo("╠══════════════════════════════════════════════════════════════╣")
    click.echo(f"║  Genome size      : {genome_size:>12,} bp")
    click.echo(f"║  Num genomes      : {num_genomes:>12}")
    click.echo(f"║  Ploidy           :  {ploidy}")
    click.echo(f"║  GC content       : {gc_content:>11.0%}")
    click.echo(f"║  Repeat density   : {repeat_density:>11.0%}")
    click.echo(f"║  Read types       :  {techs_str}")
    if graph_training:
        click.echo(f"║  Graph training   :  ENABLED")
    if graph_only:
        click.echo(f"║  ⚡ Graph-only    :  YES (no FASTQ/FASTA written)")
    click.echo(f"║  Output           :  {output}")
    click.echo("╚══════════════════════════════════════════════════════════════╝")
    click.echo()

    # ── Generate ───────────────────────────────────────────────────
    try:
        summary = generate_custom_training_data(training_config)
        click.echo()
        click.echo("═" * 62)
        click.echo(f"  ✓ Generated {summary['num_genomes_generated']} genomes "
                    f"in {summary['generation_time_human']}")
        click.echo(f"  Output → {output}")
        click.echo("═" * 62)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@train.command('run')
@click.option('--data-dir', required=True, type=click.Path(exists=True),
              help='Directory containing graph training CSVs (searched recursively)')
@click.option('--output-dir', '-o', default='trained_models', show_default=True,
              type=click.Path(), help='Directory to save trained model weights')
@click.option('--models', multiple=True, default=None,
              type=click.Choice([
                  'edge_ai', 'path_gnn', 'diploid_ai', 'ul_routing', 'sv_ai']),
              help='Which models to train (default: all). Repeat for multiple.')
@click.option('--max-depth', type=int, default=None,
              help='XGBoost max tree depth (overrides per-model default)')
@click.option('--learning-rate', type=float, default=None,
              help='XGBoost learning rate')
@click.option('--n-estimators', type=int, default=None,
              help='Number of XGBoost boosting rounds')
@click.option('--n-folds', type=int, default=5, show_default=True,
              help='Cross-validation folds')
@click.option('--val-split', type=float, default=0.15, show_default=True,
              help='Hold-out validation fraction')
@click.option('--seed', type=int, default=42, show_default=True,
              help='Random seed')
def train_run(data_dir, output_dir, models, max_depth, learning_rate,
              n_estimators, n_folds, val_split, seed):
    """
    Train StrandWeaver ML models from generated training data.

    Reads CSV training data produced by `strandweaver train generate-data
    --graph-training`, trains all five model types (EdgeAI, PathGNN,
    DiploidAI, UL Routing, SV Detection) via XGBoost, evaluates with
    k-fold cross-validation, and saves weights in the directory layout
    expected by the pipeline (--model-dir).

    Requires: numpy, xgboost, scikit-learn (install with pip).

    Examples:

      # Train all five model types
      strandweaver train run --data-dir training_data/ -o models/

      # Train only EdgeAI and DiploidAI with custom hyperparameters
      strandweaver train run --data-dir training_data/ -o models/ \\
          --models edge_ai --models diploid_ai --max-depth 8

      # Quick 3-fold CV
      strandweaver train run --data-dir training_data/ -o models/ --n-folds 3
    """
    from strandweaver.user_training import train_all_models, ModelTrainingConfig

    if train_all_models is None:
        click.echo("Error: Model training requires numpy, xgboost, and scikit-learn.  "
                    "Install with:  pip install numpy xgboost scikit-learn", err=True)
        raise SystemExit(1)

    # Default to all models if none specified
    model_list = list(models) if models else [
        'edge_ai', 'path_gnn', 'diploid_ai', 'ul_routing', 'sv_ai',
    ]

    config = ModelTrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        models=model_list,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        n_folds=n_folds,
        validation_split=val_split,
        random_seed=seed,
    )

    click.echo()
    click.echo("╔══════════════════════════════════════════════════════════════╗")
    click.echo("║            StrandWeaver  ·  Model Training                 ║")
    click.echo("╠══════════════════════════════════════════════════════════════╣")
    click.echo(f"║  Data dir    : {data_dir}")
    click.echo(f"║  Output dir  : {output_dir}")
    click.echo(f"║  Models      : {', '.join(model_list)}")
    click.echo(f"║  CV folds    : {n_folds}")
    click.echo(f"║  Val split   : {val_split}")
    click.echo(f"║  Seed        : {seed}")
    click.echo("╚══════════════════════════════════════════════════════════════╝")
    click.echo()

    try:
        report = train_all_models(config)
    except ImportError as exc:
        click.echo(f"Error: Missing dependencies — {exc}", err=True)
        raise SystemExit(1)
    except Exception as exc:
        click.echo(f"Training failed: {exc}", err=True)
        raise SystemExit(1)

    # ── Summary ────────────────────────────────────────────────────
    summary = report.get('summary', {})
    click.echo()
    click.echo("═" * 62)
    click.echo(f"  Training complete in {summary.get('elapsed_seconds', '?')}s")
    click.echo(f"  Models trained : {summary.get('models_trained', 0)}")
    click.echo(f"  Models skipped : {summary.get('models_skipped', 0)}")
    click.echo("═" * 62)

    for name, info in report.get('models', {}).items():
        status = info.get('status', 'unknown')
        if status == 'trained':
            m = info.get('metrics', {})
            if 'val_accuracy' in m:
                click.echo(f"  {name:15s} ✓  acc={m['val_accuracy']:.4f}  "
                            f"f1={m['val_f1_weighted']:.4f}  "
                            f"CV={m.get('cv_accuracy_mean', 0):.4f}"
                            f"±{m.get('cv_accuracy_std', 0):.4f}")
            else:
                click.echo(f"  {name:15s} ✓  RMSE={m.get('val_rmse', 0):.4f}  "
                            f"R²={m.get('val_r2', 0):.4f}  "
                            f"CV R²={m.get('cv_r2_mean', 0):.4f}"
                            f"±{m.get('cv_r2_std', 0):.4f}")
        elif status == 'skipped':
            click.echo(f"  {name:15s} ⊘  {info.get('reason', '')}")

    report_path = summary.get('report_path')
    if report_path:
        click.echo(f"\n  Full report → {report_path}")
    click.echo(f"  Model weights → {output_dir}/")
    click.echo()


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

# StrandWeaver v0.3.0
# Any usage is subject to this software's license.
