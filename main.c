#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read_mnist.h"

float relu(float *x, int nums) {
    for (int i = 0; i < nums; i++) {
        x[i] = x[i] < 0 ? 0 : x[i];
    }
}

float linear_layer_forward_pass(float *x, int x_rows, int x_cols, float *w,
                                int w_rows, int w_cols, float *b, float *layer_output) {
    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < w_cols; j++) {
            layer_output[i * w_cols + j] = b[j]; // Initialize with bias
            for (int k = 0; k < x_cols; k++) {
                layer_output[i * w_cols + j] += x[i * x_cols + k] * w[k * w_cols + j];
            }
        }
    }
}

void display_output(float *last_layer, int nums) {
    for (int i = 0; i < nums; i++) {
        printf("%d - %f\n", i, last_layer[i]);
    }
}

void safeSoftmaxOnlineNorm(float *inputMatix, int num_elements, float *result) {
    float m = -INFINITY;
    float new_m;
    float sum = 0;
    for (int i = 0; i < num_elements; i++) {
        new_m = fmax(inputMatix[i], m);
        sum = sum * exp(m - new_m) + exp(inputMatix[i] - new_m);
        m = new_m;
    }

    for (int i = 0; i < num_elements; i++) {
        result[i] = exp(inputMatix[i] - m) / sum;
    }
}

// Function to generate a random sample from the uniform distribution
float sample(float lower_bound, float upper_bound) {
    return lower_bound + (rand() / (RAND_MAX + 1.0)) * (upper_bound - lower_bound);
}

void w_init(float *w, int w_rows, int w_cols) {
    float k = 1 / sqrt(w_rows);
    for (int i = 0; i < w_rows * w_cols; i++) {
        w[i] = sample(-1 * k, k);
    }
}

void layer_output_reset(float *layer_output, int nums) {
    for (int i = 0; i < nums; i++) {
        layer_output[i] = 0;
    }
}

double loss_function(char y_label, float *y_preds) {
    return -log(y_preds[(int)y_label] + 1e-10);
}

// Backpropagation function
void backpropagation(float *x, int x_rows, int x_cols,
                     float *w1, int w1_rows, int w1_cols, float *b1,
                     float *w2, int w2_rows, int w2_cols, float *b2,
                     float *layer1_output, float *layer2_output, float *y_preds,
                     int y_label, float learning_rate) {

    // Calculate gradients
    float *d_loss_d_y_preds = (float *)malloc(sizeof(float) * 10);
    for (int i = 0; i < 10; i++) {
        if (i == y_label) {
            d_loss_d_y_preds[i] = y_preds[i] - 1.0f; // Cross-entropy derivative
        } else {
            d_loss_d_y_preds[i] = y_preds[i];
        }
    }

    float *d_relu2 = (float *)malloc(sizeof(float) * w2_rows * w2_cols);
    for (int i = 0; i < w2_rows * w2_cols; i++) {
        if (layer2_output[i] > 0) {
            d_relu2[i] = 1.0f;
        } else {
            d_relu2[i] = 0.0f;
        }
    }

    // Gradient of layer2 weights and bias
    float *d_loss_d_w2 = (float *)malloc(sizeof(float) * w2_rows * w2_cols);
    float *d_loss_d_b2 = (float *)malloc(sizeof(float) * w2_cols);

    for (int i = 0; i < w2_rows; i++) {
        for (int j = 0; j < w2_cols; j++) {
            d_loss_d_w2[i * w2_cols + j] = layer1_output[i] * d_loss_d_y_preds[j] * d_relu2[i * w2_cols + j];
        }
    }

    for (int j = 0; j < w2_cols; j++) {
        d_loss_d_b2[j] = d_loss_d_y_preds[j] * d_relu2[j];
    }

    // Gradient of layer1 output
    float *d_loss_d_layer1_output = (float *)malloc(sizeof(float) * w1_cols);
    for (int i = 0; i < w1_cols; i++) {
        d_loss_d_layer1_output[i] = 0.0f;
        for (int j = 0; j < w2_cols; j++) {
            d_loss_d_layer1_output[i] += w2[i * w2_cols + j] * d_loss_d_y_preds[j] * d_relu2[i * w2_cols + j];
        }
    }

    float *d_relu1 = (float *)malloc(sizeof(float) * w1_cols);
    for (int i = 0; i < w1_cols; i++) {
        if (layer1_output[i] > 0) {
            d_relu1[i] = 1.0f;
        } else {
            d_relu1[i] = 0.0f;
        }
    }

    // Gradient of layer1 weights and bias
    float *d_loss_d_w1 = (float *)malloc(sizeof(float) * w1_rows * w1_cols);
    float *d_loss_d_b1 = (float *)malloc(sizeof(float) * w1_cols);

    for (int i = 0; i < w1_rows; i++) {
        for (int j = 0; j < w1_cols; j++) {
            d_loss_d_w1[i * w1_cols + j] = x[i] * d_loss_d_layer1_output[j] * d_relu1[j];
        }
    }

    for (int j = 0; j < w1_cols; j++) {
        d_loss_d_b1[j] = d_loss_d_layer1_output[j] * d_relu1[j];
    }

    // Update weights and biases
    for (int i = 0; i < w2_rows * w2_cols; i++) {
        w2[i] -= learning_rate * d_loss_d_w2[i];
    }
    for (int i = 0; i < w2_cols; i++) {
        b2[i] -= learning_rate * d_loss_d_b2[i];
    }
    for (int i = 0; i < w1_rows * w1_cols; i++) {
        w1[i] -= learning_rate * d_loss_d_w1[i];
    }
    for (int i = 0; i < w1_cols; i++) {
        b1[i] -= learning_rate * d_loss_d_b1[i];
    }

    // Free allocated memory
    free(d_loss_d_y_preds);
    free(d_loss_d_w2);
    free(d_loss_d_b2);
    free(d_loss_d_layer1_output);
    free(d_loss_d_w1);
    free(d_loss_d_b1);
    free(d_relu1);
    free(d_relu2);
}

int main() {
    float *w1, *w2, *b1, *b2, *x;
    int w1_rows = 784, w1_cols = 128, w2_rows = 128, w2_cols = 10, x_rows = 1, x_cols = 784; // Increased neurons in layer 1 to 128

    float *layer1_output = (float *)malloc(sizeof(float) * 1 * w1_cols);
    float *layer2_output = (float *)malloc(sizeof(float) * w1_cols * w2_cols);
    float *y_preds = (float *)malloc(sizeof(float) * 10);

    w1 = (float *)malloc(sizeof(float) * w1_rows * w1_cols);
    w_init(w1, w1_rows, w1_cols);
    b1 = (float *)malloc(sizeof(float) * w1_cols);
    w_init(b1, 1, w1_cols);

    w2 = (float *)malloc(sizeof(float) * w2_rows * w2_cols);
    w_init(w2, w2_rows, w2_cols);
    b2 = (float *)malloc(sizeof(float) * w2_cols);
    w_init(b2, 1, w2_cols);

    x = (float *)malloc(sizeof(float) * 28 * 28);

    int epochs = 100;
    float learning_rate = 0.001f;

    int num_images_train, num_labels_train, num_images_test, num_labels_test;
    unsigned char **images_train = read_mnist_images("train-images.idx3-ubyte", &num_images_train);
    unsigned char *labels_train = read_mnist_labels("train-labels.idx1-ubyte", &num_labels_train);
    unsigned char **images_test = read_mnist_images("t10k-images.idx3-ubyte", &num_images_test);
    unsigned char *labels_test = read_mnist_labels("t10k-labels.idx1-ubyte", &num_labels_test);

    if (!images_train || !labels_train || !images_test || !labels_test) {
        printf("Failed to load MNIST dataset.\n");
        return 1;
    }

    printf("Number of training images: %d\n", num_images_train);
    printf("Number of training labels: %d\n", num_labels_train);
    printf("Number of testing images: %d\n", num_images_test);
    printf("Number of testing labels: %d\n", num_labels_test);

    for (int epoch = 0; epoch < epochs; epoch++) {
        double total_loss = 0.0;
        for (int d_i = 0; d_i < num_images_train; d_i++) {
            for (int i = 0; i < x_rows * x_cols; i++) {
                x[i] = (float)images_train[d_i][i] / 255.0f;
            }
            char y_label = labels_train[d_i];

            linear_layer_forward_pass(x, x_rows, x_cols, w1, w1_rows, w1_cols, b1, layer1_output);
            relu(layer1_output, x_rows * w1_cols);
            linear_layer_forward_pass(layer1_output, x_rows, w1_cols, w2, w2_rows, w2_cols, b2, layer2_output);
            relu(layer2_output, x_rows * w2_cols);
            safeSoftmaxOnlineNorm(layer2_output, 10, y_preds);
            double loss = loss_function(y_label, y_preds);
            total_loss += loss;
            backpropagation(x, x_rows, x_cols, w1, w1_rows, w1_cols, b1, w2, w2_rows, w2_cols, b2, layer1_output, layer2_output, y_preds, y_label, learning_rate);
            layer_output_reset(layer1_output, w1_cols);
            layer_output_reset(layer2_output, w2_rows * w2_cols);
        }
        printf("Epoch %d, Average Training Loss: %f\n", epoch + 1, total_loss / num_images_train);

        // Validation step
        int correct_predictions = 0;
        for (int d_i = 0; d_i < num_images_test; d_i++) {
            for (int i = 0; i < x_rows * x_cols; i++) {
                x[i] = (float)images_test[d_i][i] / 255.0f;
            }
            char y_label = labels_test[d_i];

            linear_layer_forward_pass(x, x_rows, x_cols, w1, w1_rows, w1_cols, b1, layer1_output);
            relu(layer1_output, x_rows * w1_cols);
            linear_layer_forward_pass(layer1_output, x_rows, w1_cols, w2, w2_rows, w2_cols, b2, layer2_output);
            relu(layer2_output, x_rows * w2_cols);
            safeSoftmaxOnlineNorm(layer2_output, 10, y_preds);

            int predicted_label = 0;
            float max_probability = -1.0f;
            for (int i = 0; i < 10; i++) {
                if (y_preds[i] > max_probability) {
                    max_probability = y_preds[i];
                    predicted_label = i;
                }
            }

            if (predicted_label == y_label) {
                correct_predictions++;
            }
            layer_output_reset(layer1_output, w1_cols);
            layer_output_reset(layer2_output, w2_rows * w2_cols);
        }
        float accuracy = (float)correct_predictions / num_images_test;
        printf("Epoch %d, Validation Accuracy: %f\n", epoch + 1, accuracy);
    }

    free(w1);
    free(w2);
    free(b1);
    free(b2);
    free(layer1_output);
    free(layer2_output);
    free(x);
    free(y_preds);

    for (int i = 0; i < num_images_train; i++) {
        free(images_train[i]);
    }
    free(images_train);
    free(labels_train);
    for (int i = 0; i < num_images_test; i++) {
        free(images_test[i]);
    }
    free(images_test);
    free(labels_test);

    return 0;
}