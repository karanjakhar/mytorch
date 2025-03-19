#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "read_mnist.h"

float relu(float *x, int nums){
    
    for(int i = 0; i < nums; i++){
        x[i] = x[i] < 0 ? 0 : x[i];
    }
}


float linear_layer_forward_pass(float *x, int x_rows, int x_cols, float *w, 
    int w_rows, int w_cols, float *b1, float *layer1_output){ // weight matrix shape: (in, out) -> (rows, cols)

    
    for(int i = 0; i < x_rows; i++){
        for(int j = 0; j < w_cols; j++){
            for(int k = 0; k < w_rows; k++){
                layer1_output[i*w_cols+j] += x[i*w_cols+k] * w[j*w_rows+k];
            }
        }
    }



}

float linear_layer_backward_pass(){

}


void display_output(float * last_layer, int nums){

    for(int i = 0; i < nums; i++){
        printf("%d - %f\n", i, last_layer[i]);
    }

}

void safeSoftmaxOnlineNorm(float * inputMatix,int num_elements, float * result){
    float m = -INFINITY;
    float new_m;
    float sum = 0;
    for(int i = 0; i < num_elements; i++){
        new_m = fmax(inputMatix[i], m);
        sum = sum * exp(m - new_m) + exp(inputMatix[i] - new_m);
        m = new_m;
    }

    for(int i = 0; i < num_elements; i++){
        result[i] = exp(inputMatix[i] - m)/sum;
    }
}

// Function to generate a random sample from the uniform distribution
float sample(float lower_bound, float upper_bound) {
    return lower_bound + (rand() / (RAND_MAX + 1.0)) * (upper_bound - lower_bound);
}

void w_init(float *w, int w_rows, int w_cols){

    float k = 1/sqrt(w_rows);


    for(int i = 0; i < w_rows * w_cols; i++){
        w[i] = sample(-1 * k, k);
        // printf("%f\t", w[i]);
    }

}

void layer_output_reset(float *layer_output, int nums){
    for(int i = 0; i < nums; i++){
        layer_output[i] = 0;
    }
}

int main(){

    // create weights and bias here then pass it to linear layer
    float *w1, *w2, *b1, *b2, *x;
    int w1_rows = 728, w1_cols = 4, w2_rows = 4, w2_cols = 10, x_rows = 1, x_cols = 784;

    float * layer1_output = (float *)malloc(sizeof(float) * 1 * 4);
    float * layer2_output = (float *)malloc(sizeof(float) * 4 * 10);
    float * y_preds = (float *)malloc(sizeof(float) * 10);

    w1 = (float *)malloc(sizeof(float) * 28 * 28 * 4);
    w_init(w1, w1_rows, w1_cols);
    b1 = (float *)malloc(sizeof(float) * 4);
    w_init(b1, 1, 4);
    
    w2 = (float *)malloc(sizeof(float) * 4 * 10);
    w_init(w2, w2_rows, w2_cols);
    b2 = (float *)malloc(sizeof(float) * 10);
    w_init(b2, 1, 10);

    x = (float *)malloc(sizeof(float) * 28 * 28);

    int epochs = 1;

    // let's load mnist dataset
    int num_images, num_labels;
    unsigned char** images = read_mnist_images("train-images.idx3-ubyte", &num_images);
    unsigned char* labels = read_mnist_labels("train-labels.idx1-ubyte", &num_labels);

    if (!images || !labels) {
        printf("Failed to load MNIST dataset.\n");
        return 1;
    }

    printf("Number of images: %d\n", num_images);
    printf("Number of labels: %d\n", num_labels);

    // Display the first image and its label
    // printf("Label: %d\n", labels[0]);
    // display_image(images[0]);

    
    for(int d_i=0; d_i < num_images; d_i++){
        for(int i = 0; i < x_rows*x_cols; i++){
            x[i] = (float)images[d_i][i]/255.0f;
            // printf("%f\t", x[i]);
        }
        // printf("\n");
    
   
    
        linear_layer_forward_pass(x,x_rows, x_cols, w1, w1_rows, w1_cols, b1, layer1_output);

        relu(layer1_output, x_rows * w1_cols);

        linear_layer_forward_pass(layer1_output,x_rows, w1_cols, w2, w2_rows, w2_cols, b2, layer2_output);

        relu(layer2_output, x_rows * w2_cols);

        // display_output(layer2_output, x_rows * w2_cols);

        safeSoftmaxOnlineNorm(layer2_output, 10, y_preds);

        layer_output_reset(layer1_output, 4);
        layer_output_reset(layer2_output, 40);


        printf("Output of softmax:\n");
        display_output(y_preds, 10);

    }

    free(w1);
    free(w2);
    free(b1);
    free(b2);
    free(layer1_output);
    free(layer2_output);
    free(x);
    free(y_preds);

    // Free allocated memory
    for (int i = 0; i < num_images; i++) {
        free(images[i]);
    }
    free(images);
    free(labels);

    return 0;
}
