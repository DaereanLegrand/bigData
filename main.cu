#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define MAX_WORD_LENGTH 100
#define HASH_TABLE_SIZE 10000000
#define CHUNK_SIZE (1024 * 1024 * 1024)  // 1 GB chunks
#define THREADS_PER_BLOCK 256

typedef struct {
    char word[MAX_WORD_LENGTH];
    int count;
} WordCount;

__device__ bool isAlphaDevice(char c) {
    return ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'));
}

__device__ char toLowerDevice(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c + 32;
    }
    return c;
}

__device__ unsigned int hash(const char* word) {
    unsigned int hash = 0;
    for (int i = 0; word[i] != '\0'; i++) {
        hash = 31 * hash + toLowerDevice(word[i]);
    }
    return hash % HASH_TABLE_SIZE;
}

__global__ void countWords(char* text, int textLength, WordCount* wordCounts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < textLength; i += stride) {
        if (isAlphaDevice(text[i])) {
            char word[MAX_WORD_LENGTH] = {0};
            int wordLength = 0;

            while (i < textLength && isAlphaDevice(text[i]) && wordLength < MAX_WORD_LENGTH - 1) {
                word[wordLength++] = toLowerDevice(text[i++]);
            }

            if (wordLength > 0) {
                unsigned int index = hash(word);
                atomicAdd(&wordCounts[index].count, 1);
                
                if (wordCounts[index].count == 1) {
                    for (int j = 0; j < wordLength; j++) {
                        wordCounts[index].word[j] = word[j];
                    }
                }
            }
        }
    }
}


void processFileChunk(FILE* file, char* h_text, WordCount* h_wordCounts, cudaStream_t stream) {
    size_t bytesRead = fread(h_text, 1, CHUNK_SIZE, file);
    if (bytesRead == 0) {
        printf("No bytes read or reached EOF.\n");
        return;
    }
    printf("Bytes read from file: %zu\n", bytesRead);

    char* d_text = NULL;
    WordCount* d_wordCounts = NULL;

    if (cudaMalloc(&d_text, bytesRead) != cudaSuccess) {
        printf("Failed to allocate device memory for text.\n");
        return;
    }

    if (cudaMalloc(&d_wordCounts, HASH_TABLE_SIZE * sizeof(WordCount)) != cudaSuccess) {
        printf("Failed to allocate device memory for word counts.\n");
        cudaFree(d_text);  // Free text memory before returning
        return;
    }

    if (cudaMemcpyAsync(d_text, h_text, bytesRead, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        printf("Failed to copy text to device.\n");
        cudaFree(d_text);
        cudaFree(d_wordCounts);
        return;
    }

    if (cudaMemcpyAsync(d_wordCounts, h_wordCounts, HASH_TABLE_SIZE * sizeof(WordCount), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        printf("Failed to copy word counts to device.\n");
        cudaFree(d_text);
        cudaFree(d_wordCounts);
        return;
    }

    int numBlocks = (bytesRead + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    countWords<<<numBlocks, THREADS_PER_BLOCK, 0, stream>>>(d_text, bytesRead, d_wordCounts);

    if (cudaMemcpyAsync(h_wordCounts, d_wordCounts, HASH_TABLE_SIZE * sizeof(WordCount), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        printf("Failed to copy word counts back to host.\n");
        cudaFree(d_text);
        cudaFree(d_wordCounts);
        return;
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        printf("Failed to synchronize CUDA stream.\n");
        cudaFree(d_text);
        cudaFree(d_wordCounts);
        return;
    }

    cudaFree(d_text);  // Free GPU memory after use
    cudaFree(d_wordCounts);
}

int compareWordCounts(const void* a, const void* b) {
    return ((WordCount*)b)->count - ((WordCount*)a)->count;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    FILE* file = fopen(argv[1], "r");
    if (!file) {
        printf("Error opening file: %s\n", argv[1]);
        return 1;
    }
    printf("File opened successfully: %s\n", argv[1]);

    WordCount* h_wordCounts = (WordCount*)calloc(HASH_TABLE_SIZE, sizeof(WordCount));
    if (h_wordCounts == NULL) {
        printf("Failed to allocate memory for word counts.\n");
        fclose(file);
        return 1;
    }

    char* h_text = (char*)malloc(CHUNK_SIZE);
    if (h_text == NULL) {
        printf("Failed to allocate memory for text chunk.\n");
        free(h_wordCounts);
        fclose(file);
        return 1;
    }

    cudaStream_t stream;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        printf("Failed to create CUDA stream.\n");
        free(h_text);
        free(h_wordCounts);
        fclose(file);
        return 1;
    }

    while (!feof(file)) {
        processFileChunk(file, h_text, h_wordCounts, stream);
    }

    fclose(file);
    free(h_text);

    // Sort results
    printf("Sorting results...\n");
    qsort(h_wordCounts, HASH_TABLE_SIZE, sizeof(WordCount), compareWordCounts);

    // Print top 100 words
    printf("Printing top 100 words...\n");
    for (int i = 0; i < 100 && h_wordCounts[i].count > 0; i++) {
        printf("%s: %d\n", h_wordCounts[i].word, h_wordCounts[i].count);
    }

    free(h_wordCounts);
    cudaStreamDestroy(stream);

    printf("Program completed successfully.\n");
    return 0;
}

