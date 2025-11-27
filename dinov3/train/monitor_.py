import os
import gc
import torch
import psutil
import logging
from collections import deque
from pathlib import Path

logger = logging.getLogger("dinov3")


class MemoryMonitor:
    """
    Real-time memory monitor to detect leaks during training.
    
    Usage:
        monitor = MemoryMonitor(log_every=10, alert_threshold_gb=5.0)
        
        # In training loop:
        for iteration, data in enumerate(dataloader):
            # ... training code ...
            
            monitor.log(iteration)  # Call after each iteration
    """
    
    def __init__(
        self, 
        log_every=10, 
        detailed_every=100,
        alert_threshold_gb=5.0,
        history_size=1000,
        output_file=None
    ):
        """
        Args:
            log_every: Print memory stats every N iterations
            detailed_every: Print detailed stats every N iterations
            alert_threshold_gb: Alert if memory increases by this much
            history_size: How many measurements to keep for trend analysis
            output_file: Optional CSV file to save memory history
        """
        self.log_every = log_every
        self.detailed_every = detailed_every
        self.alert_threshold_gb = alert_threshold_gb
        self.history_size = history_size
        self.output_file = output_file
        
        # Track memory over time
        self.cpu_history = deque(maxlen=history_size)
        self.gpu_history = deque(maxlen=history_size)
        self.iteration_history = deque(maxlen=history_size)
        
        # Baseline memory (set on first call)
        self.baseline_cpu = None
        self.baseline_gpu = None
        self.baseline_iteration = None
        
        # Track peaks
        self.peak_cpu = 0
        self.peak_gpu = 0
        
        # Setup CSV logging if requested
        if output_file:
            self.csv_file = Path(output_file)
            with open(self.csv_file, 'w') as f:
                f.write("iteration,cpu_gb,gpu_gb,cpu_percent,gpu_allocated_gb,gpu_reserved_gb\n")
        else:
            self.csv_file = None
    
    def get_memory_stats(self):
        """Get current memory statistics"""
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**3  # GB
        cpu_percent = process.memory_percent()
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1024**3
            gpu_reserved = torch.cuda.memory_reserved() / 1024**3
            gpu_max = torch.cuda.max_memory_allocated() / 1024**3
        else:
            gpu_allocated = 0
            gpu_reserved = 0
            gpu_max = 0
        
        return {
            'cpu_gb': cpu_mem,
            'cpu_percent': cpu_percent,
            'gpu_allocated_gb': gpu_allocated,
            'gpu_reserved_gb': gpu_reserved,
            'gpu_max_gb': gpu_max,
        }
    
    def log(self, iteration):
        """
        Log memory at current iteration.
        Call this after every training iteration.
        """
        stats = self.get_memory_stats()
        
        # Set baseline on first call
        if self.baseline_cpu is None:
            self.baseline_cpu = stats['cpu_gb']
            self.baseline_gpu = stats['gpu_allocated_gb']
            self.baseline_iteration = iteration
            logger.info(f"[Iter {iteration}] Memory baseline set: "
                       f"CPU: {self.baseline_cpu:.2f}GB, "
                       f"GPU: {self.baseline_gpu:.2f}GB")
        
        # Update history
        self.cpu_history.append(stats['cpu_gb'])
        self.gpu_history.append(stats['gpu_allocated_gb'])
        self.iteration_history.append(iteration)
        
        # Update peaks
        self.peak_cpu = max(self.peak_cpu, stats['cpu_gb'])
        self.peak_gpu = max(self.peak_gpu, stats['gpu_allocated_gb'])
        
        # Save to CSV
        if self.csv_file:
            with open(self.csv_file, 'a') as f:
                f.write(f"{iteration},{stats['cpu_gb']:.4f},{stats['gpu_allocated_gb']:.4f},"
                       f"{stats['cpu_percent']:.2f},{stats['gpu_allocated_gb']:.4f},"
                       f"{stats['gpu_reserved_gb']:.4f}\n")
        
        # Regular logging
        if iteration % self.log_every == 0:
            cpu_delta = stats['cpu_gb'] - self.baseline_cpu
            gpu_delta = stats['gpu_allocated_gb'] - self.baseline_gpu
            
            logger.info(
                f"[Iter {iteration:6d}] "
                f"CPU: {stats['cpu_gb']:6.2f}GB ({cpu_delta:+.2f}GB) | "
                f"GPU: {stats['gpu_allocated_gb']:6.2f}GB ({gpu_delta:+.2f}GB) | "
                f"GPU Reserved: {stats['gpu_reserved_gb']:6.2f}GB"
            )
            
            # Check for memory leak
            if cpu_delta > self.alert_threshold_gb:
                logger.warning(f"⚠️  CPU MEMORY LEAK DETECTED: +{cpu_delta:.2f}GB from baseline!")
            
            if gpu_delta > self.alert_threshold_gb:
                logger.warning(f"⚠️  GPU MEMORY LEAK DETECTED: +{gpu_delta:.2f}GB from baseline!")
        
        # Detailed logging
        if iteration % self.detailed_every == 0:
            self.log_detailed(iteration, stats)
        
        # Analyze trends every 1000 iterations
        if iteration % 1000 == 0 and len(self.cpu_history) > 100:
            self.analyze_trend(iteration)
    
    def log_detailed(self, iteration, stats):
        """Log detailed memory information"""
        logger.info("=" * 80)
        logger.info(f"DETAILED MEMORY REPORT - Iteration {iteration}")
        logger.info("=" * 80)
        logger.info(f"CPU Memory:")
        logger.info(f"  Current:  {stats['cpu_gb']:.2f}GB ({stats['cpu_percent']:.1f}%)")
        logger.info(f"  Baseline: {self.baseline_cpu:.2f}GB")
        logger.info(f"  Delta:    {stats['cpu_gb'] - self.baseline_cpu:+.2f}GB")
        logger.info(f"  Peak:     {self.peak_cpu:.2f}GB")
        
        if torch.cuda.is_available():
            logger.info(f"GPU Memory:")
            logger.info(f"  Allocated: {stats['gpu_allocated_gb']:.2f}GB")
            logger.info(f"  Reserved:  {stats['gpu_reserved_gb']:.2f}GB")
            logger.info(f"  Max:       {stats['gpu_max_gb']:.2f}GB")
            logger.info(f"  Baseline:  {self.baseline_gpu:.2f}GB")
            logger.info(f"  Delta:     {stats['gpu_allocated_gb'] - self.baseline_gpu:+.2f}GB")
            logger.info(f"  Peak:      {self.peak_gpu:.2f}GB")
        
        logger.info("=" * 80)
    
    def analyze_trend(self, iteration):
        """Analyze memory usage trend to detect leaks"""
        if len(self.cpu_history) < 100:
            return
        
        # Calculate average over last 100 iterations
        recent_cpu = sum(list(self.cpu_history)[-100:]) / 100
        
        # Calculate average over first 100 iterations (after baseline)
        if len(self.cpu_history) >= 200:
            early_cpu = sum(list(self.cpu_history)[:100]) / 100
            cpu_growth_rate = (recent_cpu - early_cpu) / (iteration - self.baseline_iteration) * 1000
            
            logger.info("=" * 80)
            logger.info(f"TREND ANALYSIS - Iteration {iteration}")
            logger.info("=" * 80)
            logger.info(f"CPU Memory Growth Rate: {cpu_growth_rate:.4f} GB per 1000 iterations")
            
            if cpu_growth_rate > 0.1:
                logger.warning(f"⚠️  MEMORY LEAK DETECTED!")
                logger.warning(f"   Memory is growing at {cpu_growth_rate:.4f} GB per 1000 iterations")
                logger.warning(f"   At this rate, will consume ~{cpu_growth_rate * 100:.1f}GB in 100k iterations")
            elif cpu_growth_rate > 0.01:
                logger.warning(f"⚠️  Slow memory growth detected: {cpu_growth_rate:.4f} GB/1k iters")
            else:
                logger.info(f"✓ Memory usage stable")
            
            logger.info("=" * 80)
    
    def get_summary(self):
        """Get summary of memory usage"""
        if not self.cpu_history:
            return "No data collected yet"
        
        current_cpu = self.cpu_history[-1]
        current_gpu = self.gpu_history[-1]
        
        summary = f"""
Memory Monitoring Summary
{'=' * 80}
Iterations monitored: {len(self.cpu_history)}
Baseline iteration:   {self.baseline_iteration}
Current iteration:    {self.iteration_history[-1]}

CPU Memory:
  Baseline:  {self.baseline_cpu:.2f} GB
  Current:   {current_cpu:.2f} GB
  Delta:     {current_cpu - self.baseline_cpu:+.2f} GB
  Peak:      {self.peak_cpu:.2f} GB

GPU Memory:
  Baseline:  {self.baseline_gpu:.2f} GB
  Current:   {current_gpu:.2f} GB
  Delta:     {current_gpu - self.baseline_gpu:+.2f} GB
  Peak:      {self.peak_gpu:.2f} GB
{'=' * 80}
"""
        return summary


def add_memory_monitoring_to_training(cfg, output_dir=None):
    """
    Add memory monitoring to your training loop.
    
    Returns:
        MemoryMonitor instance to use in training loop
    """
    # Setup output file
    if output_dir is None:
        output_dir = cfg.train.output_dir
    
    memory_log_file = Path(output_dir) / "memory_log.csv"
    
    # Create monitor
    monitor = MemoryMonitor(
        log_every=10,           # Print every 10 iterations
        detailed_every=100,      # Detailed report every 100 iterations
        alert_threshold_gb=5.0,  # Alert if memory grows by 5GB
        history_size=10000,      # Keep 10k iterations of history
        output_file=memory_log_file
    )
    
    logger.info(f"Memory monitoring enabled. Logs will be saved to: {memory_log_file}")
    
    return monitor


# ============================================================================
# Simple version for quick integration
# ============================================================================

def simple_memory_log(iteration, log_every=10):
    """
    Simplest possible memory logging - just add this one line to your training loop!
    
    Usage:
        for iteration, data in enumerate(dataloader):
            # ... training code ...
            simple_memory_log(iteration)
    """
    if iteration % log_every == 0:
        process = psutil.Process(os.getpid())
        cpu_mem = process.memory_info().rss / 1024**3
        
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"[Iter {iteration:6d}] CPU: {cpu_mem:6.2f}GB | GPU: {gpu_mem:6.2f}GB")
        else:
            logger.info(f"[Iter {iteration:6d}] CPU: {cpu_mem:6.2f}GB")


# ============================================================================
# Visualization (run after training or during training)
# ============================================================================

def plot_memory_usage(csv_file):
    """
    Plot memory usage over time from the CSV log.
    
    Usage:
        plot_memory_usage('output/memory_log.csv')
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("pandas and matplotlib required for plotting")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # CPU Memory
    ax1.plot(df['iteration'], df['cpu_gb'], label='CPU Memory', color='blue')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Memory (GB)')
    ax1.set_title('CPU Memory Usage Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # GPU Memory
    ax2.plot(df['iteration'], df['gpu_allocated_gb'], label='GPU Allocated', color='red')
    ax2.plot(df['iteration'], df['gpu_reserved_gb'], label='GPU Reserved', color='orange', alpha=0.7)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Memory (GB)')
    ax2.set_title('GPU Memory Usage Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    output_path = Path(csv_file).parent / 'memory_usage_plot.png'
    plt.savefig(output_path, dpi=150)
    logger.info(f"Memory plot saved to: {output_path}")
    
    # Calculate and print growth rate
    if len(df) > 1000:
        early_avg = df['cpu_gb'].iloc[:100].mean()
        late_avg = df['cpu_gb'].iloc[-100:].mean()
        growth = late_avg - early_avg
        iterations = df['iteration'].iloc[-1] - df['iteration'].iloc[0]
        growth_rate = growth / iterations * 1000
        
        print(f"\nMemory Growth Analysis:")
        print(f"  Early average: {early_avg:.2f} GB")
        print(f"  Late average:  {late_avg:.2f} GB")
        print(f"  Total growth:  {growth:.2f} GB")
        print(f"  Growth rate:   {growth_rate:.4f} GB per 1000 iterations")
        
        if growth_rate > 0.1:
            print(f"  ⚠️  SIGNIFICANT LEAK: ~{growth_rate * 100:.1f}GB in 100k iterations")
        elif growth_rate > 0.01:
            print(f"  ⚠️  Slow leak detected")
        else:
            print(f"  ✓ Memory usage stable")


# ============================================================================
# Integration Example
# ============================================================================

"""
# OPTION 1: Full featured monitoring (RECOMMENDED)
# Add to your do_train function:

def do_train(cfg, model, resume=False):
    # ... existing setup code ...
    
    # ADD THIS: Create memory monitor
    memory_monitor = add_memory_monitoring_to_training(cfg)
    
    # Training loop
    for data in metric_logger.log_every(...):
        it = iteration
        # ... all your training code ...
        
        # ADD THIS: Log memory after each iteration
        memory_monitor.log(iteration)
        
        iteration += 1
    
    # ADD THIS: Print summary at end
    logger.info(memory_monitor.get_summary())


# OPTION 2: Simplest possible (minimal code change)
# Just add one line to your training loop:

for data in metric_logger.log_every(...):
    # ... training code ...
    
    # ADD THIS ONE LINE:
    simple_memory_log(iteration, log_every=10)
    
    iteration += 1


# OPTION 3: After training, analyze the logs
plot_memory_usage('output/memory_log.csv')
"""