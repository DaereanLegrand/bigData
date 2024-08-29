import os
import sys
import multiprocessing as mp
from collections import Counter
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

# CUDA kernel for word counting
cuda_code = """
__global__ void count_words(char* text, int* word_counts, int text_len, int* word_offsets, int num_words) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_words) {
        int start = word_offsets[tid];
        int end = (tid == num_words - 1) ? text_len : word_offsets[tid + 1];
        
        // Simple hash function
        unsigned int hash = 0;
        for (int i = start; i < end; i++) {
            hash = hash * 31 + text[i];
        }
        atomicAdd(&word_counts[hash % 1000000], 1);
    }
}
"""

def process_chunk(chunk):
    words = chunk.split()
    return Counter(words)

def count_words_gpu(text):
    mod = SourceModule(cuda_code)
    count_words_kernel = mod.get_function("count_words")
    
    # Prepare data
    text_array = np.frombuffer(text.encode('ascii'), dtype=np.int8)
    word_offsets = np.array([0] + [i for i, c in enumerate(text) if c.isspace()], dtype=np.int32)
    num_words = len(word_offsets)
    
    # Allocate memory on GPU
    text_gpu = cuda.mem_alloc(text_array.nbytes)
    word_offsets_gpu = cuda.mem_alloc(word_offsets.nbytes)
    word_counts_gpu = cuda.mem_alloc(4 * 1000000)  # 4 bytes * 1,000,000 possible hash values
    
    # Copy data to GPU
    cuda.memcpy_htod(text_gpu, text_array)
    cuda.memcpy_htod(word_offsets_gpu, word_offsets)
    
    # Launch kernel
    block_size = 256
    grid_size = (num_words + block_size - 1) // block_size
    count_words_kernel(
        text_gpu,
        word_counts_gpu,
        np.int32(len(text_array)),
        word_offsets_gpu,
        np.int32(num_words),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )
    
    # Copy results back to host
    word_counts = np.zeros(1000000, dtype=np.int32)
    cuda.memcpy_dtoh(word_counts, word_counts_gpu)
    
    return word_counts

def main(filename):
    chunk_size = 1024 * 1024 * 1024  # 1 GB chunks
    total_counts = Counter()
    
    with open(filename, 'r') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            
            # Process chunk on GPU
            gpu_counts = count_words_gpu(chunk)
            
            # Merge GPU results with total counts
            for i, count in enumerate(gpu_counts):
                if count > 0:
                    total_counts[str(i)] += count
    
    # Sort and print results
    for word, count in total_counts.most_common():
        print(f"{word}: {count}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <filename>")
        sys.exit(1)
    
    filename = sys.argv[1]
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        sys.exit(1)
    
    main(filename)
