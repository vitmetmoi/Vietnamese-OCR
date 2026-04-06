"""
Optimized OCR Runner for PaddleOCR
This script provides significantly faster recognition by:
1. Increasing batch size (rec_batch_num)
2. Enabling MKL-DNN acceleration
3. Increasing CPU threads
4. Optional GPU support
"""

import json
import subprocess
import sys
import os

def run_ocr_optimized(image_path, use_gpu=False, rec_batch_num=32, cpu_threads=16, enable_mkldnn=True):
    """
    Run OCR with optimized settings for faster recognition.
    
    Args:
        image_path: Path to the image file
        use_gpu: Whether to use GPU acceleration (requires CUDA)
        rec_batch_num: Batch size for recognition (higher = faster, more memory)
        cpu_threads: Number of CPU threads to use
        enable_mkldnn: Whether to enable MKL-DNN acceleration (Intel CPUs)
    
    Returns:
        List of OCR results with labels and bounding boxes
    """
    
    command = [
        sys.executable,  # Use current Python interpreter
        "PaddleOCR/tools/infer/predict_system.py",
        "--det_algorithm=DB",
        "--det_model_dir=inference/det",
        "--rec_model_dir=inference/rec",
        f"--image_dir={image_path}",
        f"--use_gpu={'True' if use_gpu else 'False'}",
        f"--enable_mkldnn={'True' if enable_mkldnn else 'False'}",
        f"--rec_batch_num={rec_batch_num}",  # KEY OPTIMIZATION: Larger batch = faster
        f"--cpu_threads={cpu_threads}",       # More threads for better parallelism
        "--show_log=True",  # Enable logging to see timing
        "--warmup=False",
    ]
    
    print(f"Running OCR with optimizations:")
    print(f"  - rec_batch_num: {rec_batch_num}")
    print(f"  - cpu_threads: {cpu_threads}")
    print(f"  - enable_mkldnn: {enable_mkldnn}")
    print(f"  - use_gpu: {use_gpu}")
    print()
    
    # Run with subprocess for better control
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
        )
        
        # Print timing output
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"OCR Error: {result.stderr}")
            return []
            
    except Exception as e:
        print(f"OCR failed: {e}")
        return []
    
    # Load the JSON results
    try:
        with open('ocr_results.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("OCR results file not found")
        return []
    except json.JSONDecodeError:
        print("Invalid JSON in OCR results")
        return []
    
    # Transform the data
    results = []
    for item in data.get('root', []):
        points = item['points']
        results.append({
            'transcription': item['transcription'],
            'label': item['transcription'],
            'points': points,
            'bounding_box': {
                'top_left': points[0],
                'top_right': points[1],
                'bottom_right': points[2],
                'bottom_left': points[3]
            }
        })
    
    return results


def run_ocr_simple(image_path):
    """
    Simple wrapper with sensible defaults for CPU inference.
    Targets ~3-5x speedup over the original settings.
    """
    return run_ocr_optimized(
        image_path,
        use_gpu=False,
        rec_batch_num=32,      # 6 -> 32 (5x larger batches)
        cpu_threads=16,        # More threads
        enable_mkldnn=True     # Enable MKL-DNN
    )


if __name__ == "__main__":
    import time
    
    # Test with your image
    image_path = "public/11.jpeg"
    
    print("=" * 60)
    print("OPTIMIZED OCR TEST")
    print("=" * 60)
    
    start = time.time()
    ocr_results = run_ocr_simple(image_path)
    total_time = time.time() - start
    
    print(f"\n✓ Found {len(ocr_results)} text regions")
    print(f"✓ Total time: {total_time:.2f} seconds")
    
    # Print first 5 results as sample
    print("\nSample results (first 5):")
    for i, item in enumerate(ocr_results[:5], 1):
        print(f"{i}. {item['label']}")
