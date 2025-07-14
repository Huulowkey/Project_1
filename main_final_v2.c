#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include "CNN_model_forward.h"
#include <omp.h>
#include <string.h>

#define INPUT_HEIGHT 32
#define INPUT_WIDTH 32
#define INPUT_CHANNELS 3
#define NUM_CLASSES 10

// Doc du lieu train
void read_all_train_data(const char* input_file, const char* label_file, float** inputs, int** labels, int* num_samples) {
    FILE* f_input = fopen(input_file, "rb");
    FILE* f_label = fopen(label_file, "rb");
    if (!f_input || !f_label) {
        printf("Loi doc file input hoac label\n");
        exit(1);
    }

    fseek(f_input, 0, SEEK_END);
    long input_size = ftell(f_input);
    rewind(f_input);

    fseek(f_label, 0, SEEK_END);
    long label_size = ftell(f_label);
    rewind(f_label);

    *num_samples = label_size / sizeof(int);
    long expected_input_size = (long)(*num_samples) * INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float);
    if (input_size != expected_input_size) {
        printf("Loi: input_size khong khop voi so luong label\n");
        exit(1);
    }

    *inputs = (float*)malloc(input_size);
    *labels = (int*)malloc(label_size);

    if (!(*inputs) || !(*labels)) {
        printf("Khong du bo nho doc du lieu\n");
        exit(1);
    }

    fread(*inputs, 1, input_size, f_input);
    fread(*labels, 1, label_size, f_label);

    fclose(f_input);
    fclose(f_label);
}

void copy_batch(float* inputs, int* labels, float* batch_input, int* batch_label, int batch_idx, int batch_size, int input_size_per_sample) {
    int offset = batch_idx * batch_size;
    //#pragma omp parallel for
    for (int i = 0; i < batch_size; i++) {
        int src_idx = (offset + i) * input_size_per_sample;
        int dst_idx = i * input_size_per_sample;
        for (int j = 0; j < input_size_per_sample; j++) {
            batch_input[dst_idx + j] = inputs[src_idx + j];
        }
        batch_label[i] = labels[offset + i];
    }
}

int main() {
    clock_t start_program = clock();

    const int batch_size = 100;
    const int num_epochs = 10;
    const float learning_rate = 0.001f;

    float* train_inputs = NULL;
    int* train_labels = NULL;
    int num_samples = 0;

    read_all_train_data("D:/EDABK/TinyML/CNN_codeC/Input_output/train_input.bin", "D:/EDABK/TinyML/CNN_codeC/Input_output/train_label.bin", &train_inputs, &train_labels, &num_samples);

    int num_batches = num_samples / batch_size;
    int input_size_per_sample = INPUT_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;

    float* batch_input = (float*)malloc(batch_size * input_size_per_sample * sizeof(float));
    int* batch_label = (int*)malloc(batch_size * sizeof(int));
    float* grad_output_batch_input = (float*)calloc(batch_size * input_size_per_sample, sizeof(float));

    Feature_map_shape input_shape = {batch_size, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    Kernel_shape kernel_shape = {{3, 3}};
    int stride[] = {1, 1};
    int padding[] = {1, 1};
    bool use_bias = false;
    float scale;

    // Conv1
    Layer conv1;
    Params conv1_params = {NULL, NULL};
    int conv1_out_channels = 32;
    conv1_params.weight = (float*)malloc(conv1_out_channels * INPUT_CHANNELS * 3 * 3 * sizeof(float));
    scale = sqrtf(2.0f / (INPUT_CHANNELS * 3 * 3));
    #pragma omp parallel for
    for (int i = 0; i < conv1_out_channels * INPUT_CHANNELS * 3 * 3; i++) {
        conv1_params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
    float* output1 = (float*)malloc(batch_size * conv1_out_channels * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    float* grad_output1 = (float*)calloc(batch_size * conv1_out_channels * INPUT_HEIGHT * INPUT_WIDTH, sizeof(float));

    // Conv2
    Layer conv2;
    Params conv2_params = {NULL, NULL};
    int conv2_out_channels = 64;
    conv2_params.weight = (float*)malloc(conv2_out_channels * conv1_out_channels * 3 * 3 * sizeof(float));
    scale = sqrtf(2.0f / (conv1_out_channels * 3 * 3));

    #pragma omp parallel for
    for (int i = 0; i < conv2_out_channels * conv1_out_channels * 3 * 3; i++) {
        conv2_params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }

    float* output2 = (float*)malloc(batch_size * conv2_out_channels * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    float* grad_output2 = (float*)calloc(batch_size * conv2_out_channels * INPUT_HEIGHT * INPUT_WIDTH, sizeof(float));

    // BatchNorm1d
    BatchNorm1d batchnorm;
    BatchNorm1d_Params batchnorm_params = {NULL, NULL, NULL, NULL};
    float eps = 0.001f;
    float momentum = 0.99f;
    bool center = true;
    bool bscale = true;
    Feature_map_shape batchnorm_in_shape = {batch_size, conv2_out_channels, INPUT_HEIGHT, INPUT_WIDTH};
    batchnorm_params.gamma = (float*)malloc(conv2_out_channels * sizeof(float));
    batchnorm_params.beta = (float*)malloc(conv2_out_channels * sizeof(float));
    for (int i = 0; i < conv2_out_channels; i++) {
        batchnorm_params.gamma[i] = 1.0f;
        batchnorm_params.beta[i] = 0.0f;
    }
    batchnorm_params.moving_mean = (float*)calloc(conv2_out_channels, sizeof(float));
    batchnorm_params.moving_var = (float*)calloc(conv2_out_channels, sizeof(float));
    for (int i = 0; i < conv2_out_channels; i++) {
        batchnorm_params.moving_mean[i] = 0.0f;
        batchnorm_params.moving_var[i] = 1.0f;
    }
    BatchNorm1d_init(&batchnorm, batchnorm_in_shape, batchnorm_params, eps, momentum, center, bscale);
    float* output_bn = (float*)malloc(batch_size * conv2_out_channels * INPUT_HEIGHT * INPUT_WIDTH * sizeof(float));
    float* grad_output_bn = (float*)calloc(batch_size * conv2_out_channels * INPUT_HEIGHT * INPUT_WIDTH, sizeof(float));

    // Maxpool
    Layer maxpool1;
    Kernel_shape pool_size = {{2, 2}};
    int stride_mp[] = {2, 2};
    int padding_mp[] = {0, 0};
    int mp_out_height = (INPUT_HEIGHT + 2 * padding_mp[0] - pool_size.size[0]) / stride_mp[0] + 1;
    int mp_out_width = (INPUT_WIDTH + 2 * padding_mp[1] - pool_size.size[1]) / stride_mp[1] + 1;

    float* output3 = (float*)malloc(batch_size * conv2_out_channels * mp_out_height * mp_out_width * sizeof(float));
    float* grad_output3 = (float*)calloc(batch_size * conv2_out_channels * mp_out_height * mp_out_width, sizeof(float));

    int mp_size = batch_size * conv2_out_channels * mp_out_height * mp_out_width;
    maxpool1.max_indexList = (int*)malloc(mp_size * sizeof(int));
    if (!maxpool1.max_indexList) {
        printf("Khong cap phat duoc max_indexList\n");
        exit(1);
    }

    // Flatten
    float* output4 = (float*)malloc(batch_size * conv2_out_channels * mp_out_height * mp_out_width * sizeof(float));
    float* grad_output4 = (float*)calloc(batch_size * conv2_out_channels * mp_out_height * mp_out_width, sizeof(float));

    // Dense1
    int num_NodePreLayer5 = conv2_out_channels * mp_out_height * mp_out_width;
    int num_NodeThisLayer5 = 128;
    Params dense1_params = {NULL, NULL};
    dense1_params.weight = (float*)malloc(num_NodeThisLayer5 * num_NodePreLayer5 * sizeof(float));
    scale = sqrtf(2.0f / num_NodePreLayer5);
    #pragma omp parallel for
    for (int i = 0; i < num_NodeThisLayer5 * num_NodePreLayer5; i++) {
        dense1_params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
    float* output5 = (float*)malloc(batch_size * num_NodeThisLayer5 * sizeof(float));
    float* grad_output5 = (float*)calloc(batch_size * num_NodeThisLayer5, sizeof(float));

    // Dense2
    int num_NodeThisLayer6 = NUM_CLASSES;
    Params dense2_params = {NULL, NULL};
    dense2_params.weight = (float*)malloc(num_NodeThisLayer6 * num_NodeThisLayer5 * sizeof(float));
    scale = sqrtf(2.0f / num_NodeThisLayer5);
    #pragma omp parallel for
    for (int i = 0; i < num_NodeThisLayer6 * num_NodeThisLayer5; i++) {
        dense2_params.weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * scale;
    }
    float* output6 = (float*)malloc(batch_size * num_NodeThisLayer6 * sizeof(float));
    float* grad_output6 = (float*)calloc(batch_size * num_NodeThisLayer6, sizeof(float));

    int total_correct = 0;
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0;
        int correct = 0;

        double start = omp_get_wtime();
        //#pragma omp parallel for reduction(+:epoch_loss, correct)
        for (int b = 0; b < num_batches; b++) {
            //printf("\nEpoch %d/%d\n", epoch + 1, num_epochs);
            copy_batch(train_inputs, train_labels, batch_input, batch_label, b, batch_size, input_size_per_sample);

            // Forward
            Conv2d_forward(&conv1, input_shape, conv1_out_channels, kernel_shape, conv1_params, stride, padding, use_bias, batch_input, output1);
            Conv2d_forward(&conv2, Layer_get_output_shape(&conv1), conv2_out_channels, kernel_shape, conv2_params, stride, padding, use_bias, output1, output2);
            BatchNorm1d_forward(&batchnorm, output2, output_bn, true); // Training mode
            Maxpool_forward(&maxpool1, Layer_get_output_shape(&conv2), pool_size, stride_mp, padding_mp, output_bn, output3);
            Flatten_forward(output3, output4, Layer_get_output_shape(&maxpool1));
            fullyConnected_forward(num_NodePreLayer5, num_NodeThisLayer5, batch_size, dense1_params, false, output4, output5, RELU);
            fullyConnected_forward(num_NodeThisLayer5, num_NodeThisLayer6, batch_size, dense2_params, false, output5, output6, SOFTMAX);

            // Loss
            float loss = loss_sparse_categorical_crossentropy(num_NodeThisLayer6, batch_size, output6, batch_label);
            epoch_loss += loss;
            
            // Accuracy
            
            int* preds = predict_labels(output6, batch_size, num_NodeThisLayer6);
            for (int i = 0; i < batch_size; i++) {
                if (preds[i] == batch_label[i]) correct++;
            }
            free(preds);

            // Backward
            backward_sparse_categorical_crossentropy_with_softmax(num_NodeThisLayer6, batch_size, output6, batch_label, grad_output6);
            fullyConnected_backward(num_NodeThisLayer5, num_NodeThisLayer6, batch_size, dense2_params, false, output5, grad_output6, grad_output5, learning_rate);
            fullyConnected_backward(num_NodePreLayer5, num_NodeThisLayer5, batch_size, dense1_params, false, output4, grad_output5, grad_output4, learning_rate);
            FlattenBackward(grad_output4, grad_output3, Layer_get_output_shape(&maxpool1));
            Maxpool_backward(&maxpool1, grad_output3, grad_output_bn);
            BatchNorm1d_backward(&batchnorm, output2, grad_output_bn, grad_output2, learning_rate);
            Conv2d_backward(&conv2, grad_output2, grad_output1, learning_rate, output1);
            Conv2d_backward(&conv1, grad_output1, grad_output_batch_input, learning_rate, batch_input);
        }

        printf("Accuracy after epoch %d: %.2f%%\n", epoch + 1, (float)correct / num_samples * 100);
        double end = omp_get_wtime();
        printf("Training time after 1 epoch: %f seconds\n", end - start);
    }

    // // Tính độ chính xác cuối cùng trên tập huấn luyện (1000 mẫu)
    // printf("\nCalculating final accuracy on training set...\n");
    // #pragma omp parallel for reduction(+:total_correct)
    // for (int b = 0; b < num_batches; b++) {
    //     // Sao chép batch
    //     copy_batch(train_inputs, train_labels, batch_input, batch_label, b, batch_size, input_size_per_sample);

    //     // Lan truyền thuận (forward pass)
    //     Conv2d_forward(&conv1, input_shape, conv1_out_channels, kernel_shape, conv1_params, stride, padding, use_bias, batch_input, output1);
    //     Conv2d_forward(&conv2, Layer_get_output_shape(&conv1), conv2_out_channels, kernel_shape, conv2_params, stride, padding, use_bias, output1, output2);
    //     BatchNorm1d_forward(&batchnorm, output2, output_bn, false); // Chế độ suy luận
    //     Maxpool_forward(&maxpool1, Layer_get_output_shape(&conv2), pool_size, stride_mp, padding_mp, output_bn, output3);
    //     Flatten_forward(output3, output4, Layer_get_output_shape(&maxpool1));
    //     fullyConnected_forward(num_NodePreLayer5, num_NodeThisLayer5, batch_size, dense1_params, false, output4, output5, RELU);
    //     fullyConnected_forward(num_NodeThisLayer5, num_NodeThisLayer6, batch_size, dense2_params, false, output5, output6, SOFTMAX);

    //     // Dự đoán nhãn
    //     int* preds = predict_labels(output6, batch_size, num_NodeThisLayer6);
    //     if (!preds) {
    //         printf("Error: Failed to predict labels for batch %d\n", b);
    //         continue;
    //     }

    //     // So sánh nhãn dự đoán với nhãn thực tế
    //     for (int i = 0; i < batch_size; i++) {
    //         if (preds[i] == batch_label[i]) {
    //             total_correct++;
    //         }
    //     }
    //     free(preds);
    // }

    // // In độ chính xác cuối cùng
    // printf("Final training accuracy after %d epochs: %.2f%%\n", num_epochs, (float)total_correct / num_samples * 100);

    // Free memory
    free(train_inputs);
    free(train_labels);
    free(batch_input);
    free(batch_label);
    free(conv1_params.weight);
    free(conv2_params.weight);
    free(batchnorm_params.gamma);
    free(batchnorm_params.beta);
    free(batchnorm_params.moving_mean);
    free(batchnorm_params.moving_var);
    BatchNorm1d_free(&batchnorm);
    free(dense1_params.weight);
    free(dense2_params.weight);
    free(output1);
    free(output2);
    free(output_bn);
    free(output3);
    free(output4);
    free(output5);
    free(output6);
    free(grad_output1);
    free(grad_output2);
    free(grad_output_bn);
    free(grad_output3);
    free(grad_output4);
    free(grad_output5);
    free(grad_output6);
    free(grad_output_batch_input);
    free(maxpool1.max_indexList);

    printf("\nDone training!\n");
    return 0;
}