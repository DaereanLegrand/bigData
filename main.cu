#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#define HASH_TABLE_SIZE 10000000
#define CHUNK_SIZE (1024 * 1024 * 1024)
#define MAX_WORD_LENGTH 100

struct WordCount {
    char word[MAX_WORD_LENGTH];
    int count;
};

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
}

__device__ unsigned int hash(const char* str) {
    unsigned int hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c;
    return hash % HASH_TABLE_SIZE;
}

__device__ 
bool is_alpha(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

__device__ 
char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

__device__ 
void my_strcpy(char* dest, const char* src) {
    while ((*dest++ = *src++) != '\0');
}

__global__ 
void wordCounter(char* content, size_t size, WordCount* hashTable) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < size; i += stride) {
        if (is_alpha(content[i])) {
            char word[MAX_WORD_LENGTH] = {0};
            int wordLength = 0;

            while (i < size && is_alpha(content[i]) && wordLength < MAX_WORD_LENGTH - 1) {
                word[wordLength++] = to_lower(content[i++]);
            }

            if (wordLength > 0) {
                unsigned int hashValue = hash(word);
                int index = atomicAdd(&hashTable[hashValue].count, 1);
                if (index == 0) {
                    my_strcpy(hashTable[hashValue].word, word);
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return -1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file);

    char *content = (char*)malloc((fileSize + 1) * sizeof(char));
    if (content == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return -1;
    }

    size_t bytesRead = fread(content, sizeof(char), fileSize, file);
    content[bytesRead] = '\0';
    fclose(file);

    printf("Total Bytes: %lu\n", bytesRead);

    WordCount* d_hashTable;
    checkCudaErrors(cudaMalloc(&d_hashTable, HASH_TABLE_SIZE * sizeof(WordCount)));
    checkCudaErrors(cudaMemset(d_hashTable, 0, HASH_TABLE_SIZE * sizeof(WordCount)));

    int threadsPerBlock = 256;
    int blocksPerGrid = (bytesRead + threadsPerBlock - 1) / threadsPerBlock;

    for (size_t offset = 0; offset < bytesRead; offset += CHUNK_SIZE) {
        size_t chunkSize = (offset + CHUNK_SIZE > bytesRead) ? (bytesRead - offset) : CHUNK_SIZE;
        char* d_content;
        checkCudaErrors(cudaMalloc(&d_content, chunkSize));
        checkCudaErrors(cudaMemcpy(d_content, content + offset, chunkSize, cudaMemcpyHostToDevice));

        wordCounter<<<blocksPerGrid, threadsPerBlock>>>(d_content, chunkSize, d_hashTable);
        checkCudaErrors(cudaDeviceSynchronize());

        checkCudaErrors(cudaFree(d_content));
    }

    WordCount* h_hashTable = (WordCount*)malloc(HASH_TABLE_SIZE * sizeof(WordCount));
    checkCudaErrors(cudaMemcpy(h_hashTable, d_hashTable, HASH_TABLE_SIZE * sizeof(WordCount), cudaMemcpyDeviceToHost));

    thrust::device_vector<WordCount> d_wordCounts(h_hashTable, h_hashTable + HASH_TABLE_SIZE);
    thrust::sort(d_wordCounts.begin(), d_wordCounts.end(),
                 [] __device__ (const WordCount& a, const WordCount& b) { return a.count > b.count; });

    thrust::host_vector<WordCount> h_wordCounts = d_wordCounts;

    printf("Top 20 words:\n");
    for (int i = 0; i < 20 && i < h_wordCounts.size() && h_wordCounts[i].count > 0; ++i) {
        printf("%s: %d\n", h_wordCounts[i].word, h_wordCounts[i].count);
    }

    free(content);
    free(h_hashTable);
    checkCudaErrors(cudaFree(d_hashTable));

    return 0;
}
