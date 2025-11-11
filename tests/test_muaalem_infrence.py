import logging
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import psutil
import os
import statistics
import csv
from datetime import datetime


from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
import torch
from librosa.core import load

from quran_muaalem import Muaalem, MuaalemOutput, explain_for_terminal


def load_cache(cache_dir: str | Path, file_path: str | Path, reload=False):
    if reload:
        return
    file_path = Path(file_path)
    cache_path = Path(cache_dir) / f"{file_path.stem}.pt"
    if cache_path.is_file():
        print("Loading Cache")
        cache = torch.load(cache_path, weights_only=False)
        # outs = [MuaalemOutput(**item) for item in cache]
        # return outs
        return cache


def save_cache(cache_dir: str | Path, file_path: str | Path, outs: list[MuaalemOutput]):
    file_path = Path(file_path)
    cach_dir = Path(cache_dir)
    cach_dir.mkdir(exist_ok=True)
    cache_path = Path(cache_dir) / f"{file_path.stem}.pt"
    # outs_dict = [asdict(o) for o in outs]
    torch.save(outs, cache_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cache_dir = "./assets/test_cache"
    sampling_rate = 16000
    audio_path = "./assets/test_1500_ms.wav"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reload = True
    
    # Batch sizes to test: 1, 2, 4, 8, 16, 32, 64
    batch_sizes = [2**i for i in range(2)]  # [1, 2, 4, 8, 16, 32, 64]
    num_runs = 10

    uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani
    moshaf = MoshafAttributes(
        rewaya="hafs",
        madd_monfasel_len=2,
        madd_mottasel_len=4,
        madd_mottasel_waqf=4,
        madd_aared_len=2,
    )
    phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

    # Prepare CSV file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"./performance_stats_{timestamp}.csv"
    
    # CSV header
    csv_headers = [
        "batch_size",
        "avg_time_sec",
        "median_time_sec",
        "min_time_sec",
        "max_time_sec",
        "std_time_sec",
        "avg_cpu_mem_mb",
        "median_cpu_mem_mb",
        "min_cpu_mem_mb",
        "max_cpu_mem_mb",
        "std_cpu_mem_mb",
    ]
    
    if device == "cuda" and torch.cuda.is_available():
        csv_headers.extend([
            "avg_gpu_mem_mb",
            "median_gpu_mem_mb",
            "min_gpu_mem_mb",
            "max_gpu_mem_mb",
            "std_gpu_mem_mb",
            "avg_gpu_peak_mb",
            "median_gpu_peak_mb",
            "min_gpu_peak_mb",
            "max_gpu_peak_mb",
            "std_gpu_peak_mb",
        ])
    
    csv_rows = []

    # Load model once
    muaalem = Muaalem(device=device)
    wave, _ = load(audio_path, sr=sampling_rate, mono=True)
    process = psutil.Process(os.getpid())
    
    print(f"\n{'='*70}")
    print(f"Running performance tests on device: {device}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch size: {num_runs}")
    print(f"{'='*70}\n")
    
    # Loop over batch sizes
    for batch_size in batch_sizes:
        print(f"\n{'='*70}")
        print(f"Testing batch size: {batch_size}")
        print(f"{'='*70}")
        
        time_stats = []
        cpu_mem_stats = []
        gpu_mem_stats = []
        gpu_peak_stats = []
        
        for run in range(num_runs):
            # Memory profiling before inference
            mem_before = process.memory_info().rss / 1024**2  # MB
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
                gpu_mem_before = torch.cuda.memory_allocated(device) / 1024**2  # MB
            
            # Time profiling
            start_time = perf_counter()
            outs = muaalem(
                [wave for _ in range(batch_size)],
                [phonetizer_out for _ in range(batch_size)],
                sampling_rate=sampling_rate,
            )
            end_time = perf_counter()
            
            # Memory profiling after inference
            mem_after = process.memory_info().rss / 1024**2  # MB
            
            # Collect stats
            inference_time = end_time - start_time
            cpu_mem_used = mem_after - mem_before
            time_stats.append(inference_time)
            cpu_mem_stats.append(cpu_mem_used)
            
            if device == "cuda" and torch.cuda.is_available():
                gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
                gpu_mem_after = torch.cuda.memory_allocated(device) / 1024**2  # MB
                gpu_mem_used = gpu_mem_after - gpu_mem_before
                gpu_mem_stats.append(gpu_mem_used)
                gpu_peak_stats.append(gpu_mem_peak)
            
            print(f"  Run {run + 1:2d}: Time={inference_time:.4f}s, CPU Mem={cpu_mem_used:+.2f}MB", end="")
            if device == "cuda" and torch.cuda.is_available():
                print(f", GPU Mem={gpu_mem_used:.2f}MB, GPU Peak={gpu_mem_peak:.2f}MB")
            else:
                print()
        
        # Calculate and print statistics for this batch size
        avg_time = statistics.mean(time_stats)
        median_time = statistics.median(time_stats)
        min_time = min(time_stats)
        max_time = max(time_stats)
        std_time = statistics.stdev(time_stats)
        
        avg_cpu_mem = statistics.mean(cpu_mem_stats)
        median_cpu_mem = statistics.median(cpu_mem_stats)
        min_cpu_mem = min(cpu_mem_stats)
        max_cpu_mem = max(cpu_mem_stats)
        std_cpu_mem = statistics.stdev(cpu_mem_stats)
        
        print(f"\nBatch Size {batch_size} - Summary:")
        print(f"  Time:       avg={avg_time:.4f}s, median={median_time:.4f}s, min={min_time:.4f}s, max={max_time:.4f}s, std={std_time:.4f}s")
        print(f"  CPU Memory: avg={avg_cpu_mem:+.2f}MB, median={median_cpu_mem:+.2f}MB, min={min_cpu_mem:+.2f}MB, max={max_cpu_mem:+.2f}MB, std={std_cpu_mem:.2f}MB")
        
        # Prepare CSV row
        csv_row = {
            "batch_size": batch_size,
            "avg_time_sec": round(avg_time, 4),
            "median_time_sec": round(median_time, 4),
            "min_time_sec": round(min_time, 4),
            "max_time_sec": round(max_time, 4),
            "std_time_sec": round(std_time, 4),
            "avg_cpu_mem_mb": round(avg_cpu_mem, 2),
            "median_cpu_mem_mb": round(median_cpu_mem, 2),
            "min_cpu_mem_mb": round(min_cpu_mem, 2),
            "max_cpu_mem_mb": round(max_cpu_mem, 2),
            "std_cpu_mem_mb": round(std_cpu_mem, 2),
        }
        
        if device == "cuda" and torch.cuda.is_available():
            avg_gpu_mem = statistics.mean(gpu_mem_stats)
            median_gpu_mem = statistics.median(gpu_mem_stats)
            min_gpu_mem = min(gpu_mem_stats)
            max_gpu_mem = max(gpu_mem_stats)
            std_gpu_mem = statistics.stdev(gpu_mem_stats)
            
            avg_gpu_peak = statistics.mean(gpu_peak_stats)
            median_gpu_peak = statistics.median(gpu_peak_stats)
            min_gpu_peak = min(gpu_peak_stats)
            max_gpu_peak = max(gpu_peak_stats)
            std_gpu_peak = statistics.stdev(gpu_peak_stats)
            
            print(f"  GPU Memory: avg={avg_gpu_mem:.2f}MB, median={median_gpu_mem:.2f}MB, min={min_gpu_mem:.2f}MB, max={max_gpu_mem:.2f}MB, std={std_gpu_mem:.2f}MB")
            print(f"  GPU Peak:   avg={avg_gpu_peak:.2f}MB, median={median_gpu_peak:.2f}MB, min={min_gpu_peak:.2f}MB, max={max_gpu_peak:.2f}MB, std={std_gpu_peak:.2f}MB")
            
            csv_row.update({
                "avg_gpu_mem_mb": round(avg_gpu_mem, 2),
                "median_gpu_mem_mb": round(median_gpu_mem, 2),
                "min_gpu_mem_mb": round(min_gpu_mem, 2),
                "max_gpu_mem_mb": round(max_gpu_mem, 2),
                "std_gpu_mem_mb": round(std_gpu_mem, 2),
                "avg_gpu_peak_mb": round(avg_gpu_peak, 2),
                "median_gpu_peak_mb": round(median_gpu_peak, 2),
                "min_gpu_peak_mb": round(min_gpu_peak, 2),
                "max_gpu_peak_mb": round(max_gpu_peak, 2),
                "std_gpu_peak_mb": round(std_gpu_peak, 2),
            })
        
        csv_rows.append(csv_row)
    
    # Write results to CSV
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
        writer.writeheader()
        writer.writerows(csv_rows)
    
    print(f"\n{'='*70}")
    print(f"Performance results saved to: {csv_filename}")
    print(f"{'='*70}\n")
    
    # Save last output to cache
    save_cache(cache_dir, audio_path, outs)
    
    # Print sample output from last batch
    print(f"\nSample output from last batch (batch_size={batch_sizes[-1]}):")
    for i, out in enumerate(outs[:3]):  # Print first 3 outputs
        print(f"  Output {i+1}: {out.phonemes.text}")
    if len(outs) > 3:
        print(f"  ... and {len(outs) - 3} more outputs")
        
    # # Explaining Results
    # explain_for_terminal(
    #     outs[0].phonemes.text,
    #     phonetizer_out.phonemes,
    #     outs[0].sifat,
    #     phonetizer_out.sifat,
    # )
