#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <fstream>

#define MAX_WORD_LENGTH 100
#define CHUNK_SIZE (1LL * 1024 * 1024 * 1024)  // 4 GB chunks
#define MAX_WORDS_PER_BLOCK 1000000  // Estimate: 1 million words per block

struct WordLocation {
    char word[MAX_WORD_LENGTH];
    int blockId;
};

void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        exit(err);
    }
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
void wordIndexer(char* content, size_t size, int blockId, WordLocation* wordLocations, int* wordCount) {
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
                int index = atomicAdd(wordCount, 1);
                if (index < MAX_WORDS_PER_BLOCK) {
                    my_strcpy(wordLocations[index].word, word);
                    wordLocations[index].blockId = blockId;
                }
            }
        }
    }
}

size_t getAvailableGPUMemory() {
    size_t free_memory, total_memory;
    cudaMemGetInfo(&free_memory, &total_memory);
    return free_memory;
}

void processChunk(std::vector<char>& chunk, size_t chunkSize, int startBlockId, std::unordered_map<std::string, std::vector<int>>& invertedIndex) {
    size_t availableMemory = getAvailableGPUMemory();
    size_t maxBlockSize = (availableMemory * 0.8) / 2;  // Use 80% of available memory, divided between content and wordLocations

    char* d_content;
    WordLocation* d_wordLocations;
    int* d_wordCount;

    checkCudaErrors(cudaMalloc(&d_content, maxBlockSize));
    checkCudaErrors(cudaMalloc(&d_wordLocations, maxBlockSize));
    checkCudaErrors(cudaMalloc(&d_wordCount, sizeof(int)));

    for (size_t offset = 0; offset < chunkSize; offset += maxBlockSize) {
        size_t blockSize = std::min(maxBlockSize, chunkSize - offset);
        
        checkCudaErrors(cudaMemcpy(d_content, chunk.data() + offset, blockSize, cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(d_wordCount, 0, sizeof(int)));

        int threadsPerBlock = 256;
        int blocksPerGrid = (blockSize + threadsPerBlock - 1) / threadsPerBlock;

        wordIndexer<<<blocksPerGrid, threadsPerBlock>>>(d_content, blockSize, startBlockId, d_wordLocations, d_wordCount);
        checkCudaErrors(cudaDeviceSynchronize());

        int h_wordCount;
        checkCudaErrors(cudaMemcpy(&h_wordCount, d_wordCount, sizeof(int), cudaMemcpyDeviceToHost));
        h_wordCount = std::min(h_wordCount, MAX_WORDS_PER_BLOCK);

        std::vector<WordLocation> h_wordLocations(h_wordCount);
        checkCudaErrors(cudaMemcpy(h_wordLocations.data(), d_wordLocations, h_wordCount * sizeof(WordLocation), cudaMemcpyDeviceToHost));

        for (const auto& wl : h_wordLocations) {
            invertedIndex[std::string(wl.word)].push_back(startBlockId);
        }

        startBlockId++;
    }

    checkCudaErrors(cudaFree(d_content));
    checkCudaErrors(cudaFree(d_wordLocations));
    checkCudaErrors(cudaFree(d_wordCount));
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return -1;
    }

    std::ifstream file(argv[1], std::ios::binary);
    if (!file) {
        perror("Error opening file");
        return -1;
    }

    file.seekg(0, std::ios::end);
    long long fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    printf("File size: %lld bytes\n", fileSize);

    std::unordered_map<std::string, std::vector<int>> invertedIndex;
    int totalBlocks = 0;

    std::vector<char> chunk(CHUNK_SIZE);

    for (long long offset = 0; offset < fileSize; offset += CHUNK_SIZE) {
        size_t chunkSize = std::min((long long)CHUNK_SIZE, fileSize - offset);
        printf("Processing chunk at offset %lld, size %zu bytes\n", offset, chunkSize);
        
        file.read(chunk.data(), chunkSize);
        processChunk(chunk, chunkSize, totalBlocks, invertedIndex);
        totalBlocks++;
    }

    file.close();

    printf("Total chunks processed: %d\n", totalBlocks);
    printf("Total unique words: %zu\n", invertedIndex.size());

    // Query interface
    std::string queryWord;
    while (true) {
        printf("Enter a word to query (or 'quit' to exit): ");
        std::cin >> queryWord;
        if (queryWord == "quit") break;

        auto it = invertedIndex.find(queryWord);
        if (it != invertedIndex.end()) {
            printf("The word '%s' appears in %zu chunks.\n", queryWord.c_str(), it->second.size());
            printf("Chunk IDs: ");
            for (int chunkId : it->second) {
                printf("%d ", chunkId);
            }
            printf("\n");
        } else {
            printf("The word '%s' does not appear in any chunk.\n", queryWord.c_str());
        }
    }

    return 0;
}
