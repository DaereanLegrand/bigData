#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#define MAX_WORD_LENGTH 100
#define HASH_TABLE_SIZE 10000000  // Adjust based on expected unique words

typedef struct WordNode {
    char word[MAX_WORD_LENGTH];
    int count;
    struct WordNode* next;
} WordNode;

WordNode* hashTable[HASH_TABLE_SIZE] = {NULL};

unsigned int hash(const char* word) {
    unsigned int hash = 0;
    for (int i = 0; word[i] != '\0'; i++) {
        hash = 31 * hash + tolower(word[i]);
    }
    return hash % HASH_TABLE_SIZE;
}

void insertWord(const char* word) {
    unsigned int index = hash(word);
    WordNode* current = hashTable[index];

    while (current != NULL) {
        if (strcasecmp(current->word, word) == 0) {
            current->count++;
            return;
        }
        current = current->next;
    }

    WordNode* newNode = (WordNode*)malloc(sizeof(WordNode));
    strcpy(newNode->word, word);
    newNode->count = 1;
    newNode->next = hashTable[index];
    hashTable[index] = newNode;
}

void processFile(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file.\n");
        return;
    }

    char word[MAX_WORD_LENGTH];
    while (fscanf(file, "%99s", word) == 1) {
        insertWord(word);
    }

    fclose(file);
}

void printResults() {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        WordNode* current = hashTable[i];
        while (current != NULL) {
            printf("%s: %d\n", current->word, current->count);
            current = current->next;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    processFile(argv[1]);
    printResults();

    // Free allocated memory
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        WordNode* current = hashTable[i];
        while (current != NULL) {
            WordNode* temp = current;
            current = current->next;
            free(temp);
        }
    }

    return 0;
}
