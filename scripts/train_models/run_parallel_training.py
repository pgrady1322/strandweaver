#!/usr/bin/env python3
"""
Parallel Training Orchestrator with MPS GPU Acceleration

Launches multiple training jobs in parallel on Apple Silicon GPU (MPS).
- Per-technology fine-tuning heads
- Knowledge distillation (MLP and CNN students)

Features:
- Automatic MPS device detection and setup
- Process management and monitoring
- Real-time log aggregation
- Graceful error handling
- Resource monitoring
"""

import os
import sys
import subprocess
import argparse
import logging
import time
import signal
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import multiprocessing
import threading

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class TrainingJob:
    """Represents a single training job."""
    
    def __init__(
        self,
        name: str,
        script: str,
        args: Dict,
        device: str = 'mps',
        log_file: Optional[Path] = None,
    ):
        self.name = name
        self.script = script
        self.args = args
        self.device = device
        self.log_file = log_file or Path(f'/tmp/training_{name}_{int(time.time())}.log')
        self.process = None
        self.start_time = None
        self.end_time = None
        self.returncode = None
        self.logger = logging.getLogger(f"Job[{name}]")
    
    def build_command(self) -> List[str]:
        """Build command line arguments."""
        cmd = [
            'venv_arm64/bin/python3',
            self.script,
            '--device', self.device,
        ]
        
        # Add job-specific arguments
        for key, value in self.args.items():
            if key.startswith('--'):
                cmd.append(key)
                if value is not None and value is not True:
                    cmd.append(str(value))
            else:
                cmd.append(f'--{key}')
                if value is not None and value is not True:
                    cmd.append(str(value))
        
        return cmd
    
    def start(self) -> bool:
        """Start the training job."""
        cmd = self.build_command()
        self.logger.info(f"Starting job: {' '.join(cmd)}")
        
        try:
            self.start_time = datetime.now()
            self.process = subprocess.Popen(
                cmd,
                stdout=open(self.log_file, 'w'),
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.logger.info(f"Job started with PID {self.process.pid}")
            self.logger.info(f"Logs: {self.log_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start job: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if job is still running."""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for job to complete."""
        if self.process is None:
            return -1
        
        self.returncode = self.process.wait(timeout=timeout)
        self.end_time = datetime.now()
        
        if self.returncode == 0:
            self.logger.info(f"Job completed successfully in {self.elapsed_time()}")
        else:
            self.logger.error(f"Job failed with return code {self.returncode}")
        
        return self.returncode
    
    def terminate(self) -> None:
        """Terminate the job."""
        if self.process and self.is_running():
            self.logger.warning("Terminating job...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Force killing job...")
                self.process.kill()
                self.process.wait()
    
    def elapsed_time(self) -> str:
        """Get formatted elapsed time."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return f"{delta.total_seconds():.1f}s"
        elif self.start_time:
            delta = datetime.now() - self.start_time
            return f"{delta.total_seconds():.1f}s (running)"
        return "N/A"
    
    def get_status(self) -> Dict:
        """Get job status."""
        return {
            'name': self.name,
            'running': self.is_running(),
            'returncode': self.returncode,
            'elapsed_time': self.elapsed_time(),
            'log_file': str(self.log_file),
        }


class ParallelTrainer:
    """Orchestrates parallel training jobs."""
    
    def __init__(
        self,
        repo_root: Path,
        output_dir: Path,
        device: str = 'mps',
        max_parallel_jobs: int = 3,
    ):
        self.repo_root = repo_root
        self.output_dir = output_dir
        self.device = device
        self.max_parallel_jobs = max_parallel_jobs
        self.jobs: Dict[str, TrainingJob] = {}
        self.logger = logging.getLogger('ParallelTrainer')
        self.start_time = None
        self.end_time = None
        
        # Setup output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.output_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    def create_tech_heads_job(
        self,
        baseline_model: Path,
        training_data: Path,
        heads_output: Path,
        epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
    ) -> TrainingJob:
        """Create per-technology heads training job."""
        args = {
            '--baseline': str(baseline_model),
            '--data': str(training_data),
            '--output': str(heads_output),
            '--epochs': epochs,
            '--batch-size': batch_size,
            '--learning-rate': learning_rate,
        }
        
        script = str(
            self.repo_root / 'scripts' / 'train_models' / 'train_tech_specific_heads.py'
        )
        
        log_file = self.logs_dir / 'tech_heads_training.log'
        
        return TrainingJob(
            name='tech_heads',
            script=script,
            args=args,
            device=self.device,
            log_file=log_file,
        )
    
    def create_distillation_job(
        self,
        teacher_model: Path,
        student_type: str,
        training_data: Path,
        student_output: Path,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        temperature: float = 4.0,
        alpha: float = 0.3,
    ) -> TrainingJob:
        """Create knowledge distillation job."""
        args = {
            '--teacher': str(teacher_model),
            '--student-type': student_type,
            '--data': str(training_data),
            '--output': str(student_output),
            '--epochs': epochs,
            '--batch-size': batch_size,
            '--learning-rate': learning_rate,
            '--temperature': temperature,
            '--alpha': alpha,
        }
        
        script = str(
            self.repo_root / 'scripts' / 'train_models' / 'train_knowledge_distillation.py'
        )
        
        log_file = self.logs_dir / f'distillation_{student_type}_training.log'
        
        return TrainingJob(
            name=f'distillation_{student_type}',
            script=script,
            args=args,
            device=self.device,
            log_file=log_file,
        )
    
    def add_job(self, job: TrainingJob) -> None:
        """Add job to the queue."""
        self.jobs[job.name] = job
        self.logger.info(f"Added job: {job.name}")
    
    def run_all(self, wait: bool = True) -> Dict:
        """Run all jobs in parallel (up to max_parallel_jobs at a time)."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"STARTING PARALLEL TRAINING ({len(self.jobs)} jobs)")
        self.logger.info(f"{'='*70}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Max parallel: {self.max_parallel_jobs}")
        
        self.start_time = datetime.now()
        
        # Start all jobs
        active_jobs = []
        pending_jobs = list(self.jobs.values())
        
        # Signal handler for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.warning("\nReceived interrupt signal, shutting down...")
            self.terminate_all()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            while pending_jobs or active_jobs:
                # Start new jobs if we have capacity
                while len(active_jobs) < self.max_parallel_jobs and pending_jobs:
                    job = pending_jobs.pop(0)
                    job.start()
                    active_jobs.append(job)
                
                # Check for completed jobs
                still_running = []
                for job in active_jobs:
                    if job.is_running():
                        still_running.append(job)
                    else:
                        self.logger.info(f"Job completed: {job.name}")
                
                active_jobs = still_running
                
                # Print status
                if active_jobs:
                    self.logger.info(f"Active jobs ({len(active_jobs)}/{self.max_parallel_jobs}):")
                    for job in active_jobs:
                        self.logger.info(f"  - {job.name}: {job.elapsed_time()} [PID {job.process.pid}]")
                    
                    # Wait before checking again
                    time.sleep(5)
            
            self.end_time = datetime.now()
            
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user")
            self.terminate_all()
            raise
        
        # Return results
        return self.get_results()
    
    def terminate_all(self) -> None:
        """Terminate all jobs."""
        self.logger.warning("Terminating all jobs...")
        for job in self.jobs.values():
            if job.is_running():
                job.terminate()
    
    def get_results(self) -> Dict:
        """Get training results summary."""
        total_time = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        
        results = {
            'total_time': total_time,
            'jobs': {},
        }
        
        for name, job in self.jobs.items():
            results['jobs'][name] = {
                'name': job.name,
                'success': job.returncode == 0,
                'returncode': job.returncode,
                'elapsed_time': job.elapsed_time(),
                'log_file': str(job.log_file),
            }
        
        return results
    
    def print_summary(self, results: Dict) -> None:
        """Print training summary."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info("PARALLEL TRAINING SUMMARY")
        self.logger.info(f"{'='*70}")
        
        for job_name, job_result in results['jobs'].items():
            status = "✅ SUCCESS" if job_result['success'] else "❌ FAILED"
            self.logger.info(
                f"{status}: {job_name:30} ({job_result['elapsed_time']})"
            )
        
        total_time = results['total_time']
        self.logger.info(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
        
        # Save detailed results
        results_file = self.output_dir / 'training_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to: {results_file}")
    
    def tail_logs(self, job_name: str, lines: int = 20) -> None:
        """Print last N lines of a job's log."""
        if job_name not in self.jobs:
            self.logger.error(f"Job not found: {job_name}")
            return
        
        job = self.jobs[job_name]
        if not job.log_file.exists():
            self.logger.error(f"Log file not found: {job.log_file}")
            return
        
        self.logger.info(f"\nLast {lines} lines of {job_name}:")
        self.logger.info(f"{'='*70}")
        
        with open(job.log_file, 'r') as f:
            all_lines = f.readlines()
            for line in all_lines[-lines:]:
                print(line.rstrip())
        
        self.logger.info(f"{'='*70}")


def get_device_info() -> str:
    """Get information about available devices."""
    try:
        import torch
        
        info_lines = []
        
        # Check MPS
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            info_lines.append("✅ MPS (Apple Silicon) available")
        else:
            info_lines.append("❌ MPS (Apple Silicon) not available")
        
        # Check CUDA
        if torch.cuda.is_available():
            info_lines.append(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            info_lines.append("❌ CUDA not available")
        
        info_lines.append(f"CPU count: {multiprocessing.cpu_count()}")
        
        return '\n'.join(info_lines)
    except ImportError:
        return "PyTorch not installed"


def main():
    parser = argparse.ArgumentParser(
        description='Parallel training orchestrator with MPS GPU acceleration'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'tech-heads', 'distill-mlp', 'distill-cnn', 'distill-both'],
        default='all',
        help='Training mode: all (default), tech-heads only, or distillation options',
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='models/error_predictor_v2.pt',
        help='Path to baseline model',
    )
    parser.add_argument(
        '--data',
        type=str,
        default='training_data/read_correction_v2/base_error',
        help='Path to training data',
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models',
        help='Output directory for trained models',
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['mps', 'cuda', 'cpu', 'auto'],
        default='mps',
        help='Device to use (auto-detect if mps not available)',
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=3,
        help='Maximum number of parallel jobs',
    )
    parser.add_argument(
        '--epochs-heads',
        type=int,
        default=20,
        help='Epochs for per-tech heads training',
    )
    parser.add_argument(
        '--epochs-distill',
        type=int,
        default=50,
        help='Epochs for knowledge distillation',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training',
    )
    parser.add_argument(
        '--tail',
        type=str,
        help='Print last 20 lines of specified job log (e.g., tech_heads)',
    )
    
    args = parser.parse_args()
    
    # Get repo root
    repo_root = Path(__file__).parent.parent.parent
    output_dir = repo_root / args.output
    
    logger.info(f"\n{'='*70}")
    logger.info("PARALLEL TRAINING SETUP")
    logger.info(f"{'='*70}")
    logger.info(f"Repository: {repo_root}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nDevice information:")
    logger.info(get_device_info())
    
    # Determine actual device
    device = args.device
    if device == 'auto':
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
                logger.info("Auto-detected MPS device")
            elif torch.cuda.is_available():
                device = 'cuda'
                logger.info("Auto-detected CUDA device")
            else:
                device = 'cpu'
                logger.warning("No GPU detected, using CPU")
        except ImportError:
            device = 'cpu'
            logger.warning("PyTorch not installed, using CPU")
    
    # Verify files exist
    baseline_path = repo_root / args.baseline
    data_path = repo_root / args.data
    
    if not baseline_path.exists():
        logger.error(f"Baseline model not found: {baseline_path}")
        sys.exit(1)
    
    if not data_path.exists():
        logger.error(f"Training data not found: {data_path}")
        sys.exit(1)
    
    logger.info(f"✓ Baseline model: {baseline_path}")
    logger.info(f"✓ Training data: {data_path}")
    
    # If tail mode, handle it
    if args.tail:
        trainer = ParallelTrainer(repo_root, output_dir, device=device)
        trainer.tail_logs(args.tail)
        return
    
    # Create trainer
    trainer = ParallelTrainer(
        repo_root=repo_root,
        output_dir=output_dir,
        device=device,
        max_parallel_jobs=args.max_parallel,
    )
    
    # Add jobs based on mode
    heads_output = output_dir / 'tech_specific_heads'
    student_output = output_dir / 'student_models'
    
    if args.mode in ['all', 'tech-heads']:
        job = trainer.create_tech_heads_job(
            baseline_model=baseline_path,
            training_data=data_path,
            heads_output=heads_output,
            epochs=args.epochs_heads,
            batch_size=args.batch_size,
        )
        trainer.add_job(job)
    
    if args.mode in ['all', 'distill-mlp', 'distill-both']:
        job = trainer.create_distillation_job(
            teacher_model=baseline_path,
            student_type='mlp',
            training_data=data_path,
            student_output=student_output,
            epochs=args.epochs_distill,
            batch_size=args.batch_size,
        )
        trainer.add_job(job)
    
    if args.mode in ['all', 'distill-cnn', 'distill-both']:
        job = trainer.create_distillation_job(
            teacher_model=baseline_path,
            student_type='cnn',
            training_data=data_path,
            student_output=student_output,
            epochs=args.epochs_distill,
            batch_size=args.batch_size,
        )
        trainer.add_job(job)
    
    if not trainer.jobs:
        logger.error("No jobs to run")
        sys.exit(1)
    
    logger.info(f"\nRunning {len(trainer.jobs)} jobs in parallel (max {args.max_parallel})...")
    
    # Run training
    try:
        results = trainer.run_all(wait=True)
        trainer.print_summary(results)
        
        # Check for failures
        failed = sum(1 for j in results['jobs'].values() if not j['success'])
        if failed > 0:
            logger.warning(f"{failed} job(s) failed")
            sys.exit(1)
        else:
            logger.info("✅ All jobs completed successfully!")
            sys.exit(0)
    
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)


if __name__ == '__main__':
    main()
