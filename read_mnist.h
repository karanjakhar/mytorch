#ifndef READ_MNIST
#define READ_MNIST


unsigned char** read_mnist_images(const char* filename, int* num_images);
unsigned char* read_mnist_labels(const char* filename, int* num_labels);
void display_image(unsigned char* image);

#endif