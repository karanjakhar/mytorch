#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define IMAGE_SIZE 28 * 28

// Function to swap endianess (MNIST files are big-endian)
static uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) | ((val << 8) & 0xff0000) |
           ((val >> 8) & 0xff00) | ((val << 24) & 0xff000000);
}

// Function to read the MNIST images
unsigned char** read_mnist_images(const char* filename, int* num_images) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    uint32_t magic, count, rows, cols;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&count, sizeof(uint32_t), 1, file);
    fread(&rows, sizeof(uint32_t), 1, file);
    fread(&cols, sizeof(uint32_t), 1, file);

    magic = swap_endian(magic);
    count = swap_endian(count);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    if (magic != 2051) {
        printf("Invalid magic number in image file: %u\n", magic);
        fclose(file);
        return NULL;
    }

    *num_images = count;
    unsigned char** images = (unsigned char**)malloc(count * sizeof(unsigned char*));
    for (int i = 0; i < count; i++) {
        images[i] = (unsigned char*)malloc(rows * cols * sizeof(unsigned char));
        fread(images[i], sizeof(unsigned char), rows * cols, file);
    }

    fclose(file);
    return images;
}

// Function to read the MNIST labels
unsigned char* read_mnist_labels(const char* filename, int* num_labels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file: %s\n", filename);
        return NULL;
    }

    uint32_t magic, count;
    fread(&magic, sizeof(uint32_t), 1, file);
    fread(&count, sizeof(uint32_t), 1, file);

    magic = swap_endian(magic);
    count = swap_endian(count);

    if (magic != 2049) {
        printf("Invalid magic number in label file: %u\n", magic);
        fclose(file);
        return NULL;
    }

    *num_labels = count;
    unsigned char* labels = (unsigned char*)malloc(count * sizeof(unsigned char));
    fread(labels, sizeof(unsigned char), count, file);

    fclose(file);
    return labels;
}

// Function to display an image
void display_image(unsigned char* image) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%c", image[i * 28 + j] > 128 ? '#' : ' ');
        }
        printf("\n");
    }
}

// int main() {
//     int num_images, num_labels;
//     unsigned char** images = read_mnist_images("train-images.idx3-ubyte", &num_images);
//     unsigned char* labels = read_mnist_labels("train-labels.idx1-ubyte", &num_labels);

//     if (!images || !labels) {
//         printf("Failed to load MNIST dataset.\n");
//         return 1;
//     }

//     printf("Number of images: %d\n", num_images);
//     printf("Number of labels: %d\n", num_labels);

//     // Display the first image and its label
//     printf("Label: %d\n", labels[0]);
//     display_image(images[0]);

//     // Free allocated memory
//     for (int i = 0; i < num_images; i++) {
//         free(images[i]);
//     }
//     free(images);
//     free(labels);

//     return 0;
// }
