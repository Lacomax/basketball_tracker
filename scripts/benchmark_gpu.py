#!/usr/bin/env python3
"""
GPU Performance Benchmark for Basketball Tracker

Tests different configurations to find optimal settings for your RTX GPU.
Measures:
- Training throughput (images/second)
- Inference speed (FPS)
- Memory usage
- Different batch sizes and precision modes

Usage:
    python scripts/benchmark_gpu.py [--quick]
"""

import torch
import time
import psutil
import numpy as np
from ultralytics import YOLO
import cv2
import sys
import os


def print_separator(title=""):
    """Print a separator line."""
    if title:
        print(f"\n{'='*70}")
        print(f"{title:^70}")
        print(f"{'='*70}")
    else:
        print(f"{'-'*70}")


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated(0) / 1024**3,
            'reserved': torch.cuda.memory_reserved(0) / 1024**3,
            'total': torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    return None


def print_gpu_info():
    """Print detailed GPU information."""
    print_separator("GPU INFORMATION")

    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        print("\nTo install PyTorch with CUDA support:")
        print("conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False

    print(f"✓ CUDA Available")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  PyTorch Version: {torch.__version__}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Multi-Processors: {props.multi_processor_count}")

        # Determine GPU generation and features
        if props.major >= 8:  # Ampere (RTX 30xx), Ada (RTX 40xx)
            print(f"  Architecture: Ampere/Ada (RTX 30xx/40xx series)")
            print(f"  Tensor Cores: Gen 3/4 ✓")
            print(f"  FP16 Speedup: ~2-3x")
            print(f"  Recommended: Use FP16 + large batch sizes")
        elif props.major == 7 and props.minor >= 5:  # Turing (RTX 20xx)
            print(f"  Architecture: Turing (RTX 20xx series)")
            print(f"  Tensor Cores: Gen 2 ✓")
            print(f"  FP16 Speedup: ~2x")
            print(f"  Recommended: Use FP16")
        elif props.major == 7:  # Volta
            print(f"  Architecture: Volta")
            print(f"  Tensor Cores: Gen 1 ✓")
        else:
            print(f"  Tensor Cores: ✗")
            print(f"  Warning: FP16 may not provide speedup")

    mem = get_gpu_memory()
    print(f"\nCurrent Memory Usage:")
    print(f"  Allocated: {mem['allocated']:.2f} GB")
    print(f"  Reserved: {mem['reserved']:.2f} GB")
    print(f"  Available: {mem['total'] - mem['reserved']:.2f} GB")

    return True


def benchmark_inference(model_path='yolo11n.pt', batch_sizes=[1, 4, 8, 16], num_frames=100, imgsz=640):
    """Benchmark inference speed with different batch sizes and precision modes."""
    print_separator("INFERENCE BENCHMARK")

    # Create dummy frames
    print(f"Generating {num_frames} test frames ({imgsz}x{imgsz})...")
    frames = [np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(num_frames)]

    results = []

    for batch_size in batch_sizes:
        for use_fp16 in [False, True]:
            precision = "FP16" if use_fp16 else "FP32"

            print(f"\nTesting: Batch={batch_size}, Precision={precision}")

            try:
                # Load model
                model = YOLO(model_path)

                # Move to GPU
                if torch.cuda.is_available():
                    model.to('cuda:0')

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                mem_before = get_gpu_memory()

                # Warmup
                _ = model(frames[0], half=use_fp16, verbose=False, device=0)

                # Benchmark
                start_time = time.time()
                processed_frames = 0

                for i in range(0, num_frames, batch_size):
                    batch = frames[i:i+batch_size]
                    _ = model(batch, half=use_fp16, verbose=False, device=0)
                    processed_frames += len(batch)

                # Sync GPU
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed = time.time() - start_time
                fps = processed_frames / elapsed

                mem_after = get_gpu_memory()
                mem_used = mem_after['allocated'] - mem_before['allocated']

                print(f"  ✓ FPS: {fps:.1f}")
                print(f"  ✓ Latency: {1000/fps:.1f} ms/frame")
                print(f"  ✓ Memory: {mem_used*1024:.0f} MB")
                print(f"  ✓ Time: {elapsed:.2f}s for {processed_frames} frames")

                results.append({
                    'batch_size': batch_size,
                    'precision': precision,
                    'fps': fps,
                    'latency_ms': 1000/fps,
                    'memory_mb': mem_used*1024,
                    'total_time': elapsed
                })

                # Cleanup
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue

    # Print summary
    print_separator("INFERENCE SUMMARY")
    print(f"{'Batch':<8} {'Precision':<10} {'FPS':<10} {'Latency':<12} {'Memory':<12}")
    print("-" * 70)

    for r in results:
        print(f"{r['batch_size']:<8} {r['precision']:<10} {r['fps']:<10.1f} "
              f"{r['latency_ms']:<12.1f} {r['memory_mb']:<12.0f}")

    # Find best configuration
    if results:
        best_fps = max(results, key=lambda x: x['fps'])
        print(f"\n🏆 Best Performance:")
        print(f"   Batch={best_fps['batch_size']}, Precision={best_fps['precision']}")
        print(f"   {best_fps['fps']:.1f} FPS ({best_fps['latency_ms']:.1f} ms/frame)")

    return results


def benchmark_training(epochs=3, batch_sizes=[8, 16, 24, 32], imgsz=640):
    """Benchmark training speed with different batch sizes."""
    print_separator("TRAINING BENCHMARK")

    # Check if we have a dataset
    if not os.path.exists('data/basketball_combined/data.yaml'):
        print("⚠️  No training dataset found at data/basketball_combined/")
        print("   Skipping training benchmark")
        print("   Run this after combining datasets")
        return []

    print(f"Testing training with {epochs} epochs, {imgsz}x{imgsz} images")

    results = []

    for batch_size in batch_sizes:
        for use_fp16 in [False, True]:
            precision = "AMP" if use_fp16 else "FP32"

            print(f"\nTesting: Batch={batch_size}, Precision={precision}")

            try:
                # Load model
                model = YOLO('yolo11n.pt')  # Use nano for quick benchmark

                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                mem_before = get_gpu_memory()
                start_time = time.time()

                # Train
                model.train(
                    data='data/basketball_combined/data.yaml',
                    epochs=epochs,
                    batch=batch_size,
                    imgsz=imgsz,
                    device=0,
                    amp=use_fp16,
                    verbose=False,
                    project='runs/benchmark',
                    name=f'batch{batch_size}_{precision}',
                    exist_ok=True,
                )

                elapsed = time.time() - start_time
                mem_after = get_gpu_memory()
                mem_used = mem_after['reserved']  # Use reserved for training

                # Calculate throughput
                # Assume ~100 images per epoch (rough estimate)
                images_per_epoch = 100
                total_images = images_per_epoch * epochs
                throughput = total_images / elapsed

                print(f"  ✓ Throughput: {throughput:.1f} images/sec")
                print(f"  ✓ Time: {elapsed:.1f}s for {epochs} epochs")
                print(f"  ✓ Memory: {mem_used*1024:.0f} MB")

                results.append({
                    'batch_size': batch_size,
                    'precision': precision,
                    'throughput': throughput,
                    'total_time': elapsed,
                    'memory_mb': mem_used*1024,
                })

                # Cleanup
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"  ❌ Out of Memory")
                else:
                    print(f"  ❌ Failed: {e}")
                continue
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                continue

    # Print summary
    if results:
        print_separator("TRAINING SUMMARY")
        print(f"{'Batch':<8} {'Precision':<10} {'Throughput':<15} {'Time':<12} {'Memory':<12}")
        print("-" * 70)

        for r in results:
            print(f"{r['batch_size']:<8} {r['precision']:<10} {r['throughput']:<15.1f} "
                  f"{r['total_time']:<12.1f} {r['memory_mb']:<12.0f}")

        best = max(results, key=lambda x: x['throughput'])
        print(f"\n🏆 Best Performance:")
        print(f"   Batch={best['batch_size']}, Precision={best['precision']}")
        print(f"   {best['throughput']:.1f} images/sec")

    return results


def recommend_settings():
    """Recommend optimal settings based on GPU."""
    print_separator("RECOMMENDED SETTINGS")

    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected")
        return

    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1024**3

    print(f"Based on your GPU: {props.name} ({mem_gb:.0f} GB)")
    print()

    # Categorize GPU
    if mem_gb >= 20:  # RTX 4090, 3090, etc.
        print("🚀 High-End GPU Detected")
        print("\nRecommended Training Settings:")
        print("  preset: max_performance")
        print("  batch_size: 48-64")
        print("  workers: 8")
        print("  model: yolo11l.pt or yolo11x.pt")
        print("  amp: true")
        print("  cache: true")
        print("\nRecommended Inference Settings:")
        print("  half: true")
        print("  batch_size: 16")
        print("  use_tensorrt: true (after export)")

    elif mem_gb >= 10:  # RTX 4070, 3080, 3070, etc.
        print("⚡ Mid-Range GPU Detected")
        print("\nRecommended Training Settings:")
        print("  preset: balanced")
        print("  batch_size: 24-32")
        print("  workers: 6")
        print("  model: yolo11l.pt")
        print("  amp: true")
        print("  cache: true")
        print("\nRecommended Inference Settings:")
        print("  half: true")
        print("  batch_size: 8-12")

    else:  # RTX 3060, etc.
        print("💻 Entry-Level GPU Detected")
        print("\nRecommended Training Settings:")
        print("  preset: memory_efficient")
        print("  batch_size: 12-16")
        print("  workers: 4")
        print("  model: yolo11m.pt or yolo11l.pt")
        print("  amp: true")
        print("  cache: 'disk'")
        print("\nRecommended Inference Settings:")
        print("  half: true")
        print("  batch_size: 4-6")

    print("\nUpdate these settings in: config_performance.yaml")


def main():
    """Main benchmark function."""
    import argparse

    parser = argparse.ArgumentParser(description='GPU Performance Benchmark')
    parser.add_argument('--quick', action='store_true',
                       help='Quick benchmark (fewer iterations)')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training benchmark')
    parser.add_argument('--model', default='yolo11n.pt',
                       help='Model to benchmark (default: yolo11n.pt)')
    args = parser.parse_args()

    print("="*70)
    print("BASKETBALL TRACKER - GPU PERFORMANCE BENCHMARK")
    print("="*70)

    # System info
    print_separator("SYSTEM INFORMATION")
    print(f"CPU: {psutil.cpu_count(logical=False)} cores ({psutil.cpu_count(logical=True)} threads)")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    print(f"Python: {sys.version.split()[0]}")

    # GPU info
    if not print_gpu_info():
        return 1

    # Recommendations
    recommend_settings()

    # Run benchmarks
    if args.quick:
        print("\n🏃 Running QUICK benchmark...")
        batch_sizes = [8, 16]
        num_frames = 50
        epochs = 1
    else:
        print("\n🔬 Running FULL benchmark (this may take a while)...")
        batch_sizes = [4, 8, 16, 24, 32]
        num_frames = 100
        epochs = 3

    # Inference benchmark
    print("\n")
    inf_results = benchmark_inference(
        model_path=args.model,
        batch_sizes=batch_sizes,
        num_frames=num_frames
    )

    # Training benchmark
    if not args.skip_training:
        print("\n")
        train_results = benchmark_training(
            epochs=epochs,
            batch_sizes=batch_sizes[:3]  # Only test smaller batch sizes
        )

    print_separator("BENCHMARK COMPLETE")
    print("\nNext steps:")
    print("1. Update config_performance.yaml with optimal settings")
    print("2. Run: python scripts/train_basketball_detector_optimized.py")
    print("3. For max inference speed, export model to TensorRT")

    return 0


if __name__ == '__main__':
    sys.exit(main())
