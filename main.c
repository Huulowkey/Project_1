#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "CNN_model.h"
#include <time.h>

int main() {
    // Đo thời gian toàn bộ chương trình
    clock_t start_program = clock();

    int batch_size = 500;
    //*************************************************************layer 1****************************************************************
    Feature_map_shape input_shape = {batch_size, 3, 32, 32};
    Kernel_shape kernel_shape = {{3, 3}}; 
    Params params1 = {NULL, NULL};
    
    // Parameters for Conv2d
    int output_channels_1 = 32;
    int padding[] = {1, 1};
    int stride[] = {1, 1};

    Layer conv_layer_1;
    bool use_bias = false;
    
    // Load weight 1
    char conv1_weight[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/conv1_weights.bin";
    readData_Conv2D_FromFile(conv1_weight, &params1.weight);
    if (!params1.weight) {
        printf("Error: Failed to load conv1 weights\n");
        return 1;
    }
    
    // Allocate input and output for conv layer
    float* input = (float*)malloc(input_shape.batch_size * input_shape.channels * input_shape.height * input_shape.width * sizeof(float));
    if (!input) {
        printf("Error: Failed to allocate memory for input\n");
        free(params1.weight);
        return 1;
    }
    float* output1 = (float*)malloc(input_shape.batch_size * output_channels_1 * input_shape.height * input_shape.width * sizeof(float));
    if (!output1) {
        printf("Error: Failed to allocate memory for output1\n");
        free(input);
        free(params1.weight);
        return 1;
    }
    
    // Load input
    char file_input[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/input.bin";   
    readData_Conv2D_FromFile(file_input, &input);
    if (!input) {
        printf("Error: Failed to load input data\n");
        free(output1);
        free(params1.weight);
        return 1;
    }
    
    // Run forward pass for conv layer
    Conv2d_forward(&conv_layer_1, input_shape, output_channels_1, kernel_shape, params1, stride, padding, use_bias, input, output1);

    // Save conv output
    char file_output1[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/conv1_output.bin";
    saveOutput_Conv2D_ToFile(file_output1, output1, input_shape.batch_size, output_channels_1, conv_layer_1.out_shape.height, conv_layer_1.out_shape.width);
    
    // Print output shapes
    printf("After conv2d_1: (%d, %d, %d, %d)\n", 
           conv_layer_1.out_shape.batch_size, 
           conv_layer_1.out_shape.channels, 
           conv_layer_1.out_shape.height, 
           conv_layer_1.out_shape.width);
    free(input);
    free(params1.weight);
    //*************************************************************layer 2****************************************************************
    Feature_map_shape input2_shape = conv_layer_1.out_shape;
    Params params2 = {NULL, NULL};
    int output_channels_2 = 64;
    Layer conv_layer_2;
    
    float* output2 = (float*)malloc(input2_shape.batch_size * output_channels_2 * input2_shape.height * input2_shape.width * sizeof(float));
    if (!output2) {
        printf("Error: Failed to allocate memory for output2\n");
        free(output1);
        return 1;
    }
    // Load weight 2
    char conv2_weight[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/conv2_weights.bin";
    readData_Conv2D_FromFile(conv2_weight, &params2.weight);
    if (!params2.weight) {
        printf("Error: Failed to load conv2 weights\n");
        free(output1);
        free(output2);
        return 1;
    }
    // Run forward pass for conv layer
    Conv2d_forward(&conv_layer_2, input2_shape, output_channels_2, kernel_shape, params2, stride, padding, use_bias, output1, output2);
    // Save conv output
    char file_output2[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/conv2_output.bin";
    saveOutput_Conv2D_ToFile(file_output2, output2, input2_shape.batch_size, output_channels_2, conv_layer_2.out_shape.height, conv_layer_2.out_shape.width);
    // Print output shapes
    printf("After conv2d_2: (%d, %d, %d, %d)\n", 
           conv_layer_2.out_shape.batch_size, 
           conv_layer_2.out_shape.channels, 
           conv_layer_2.out_shape.height, 
           conv_layer_2.out_shape.width);
    free(output1);
    free(params2.weight);
    //*************************************************************layer 3****************************************************************
    Feature_map_shape input3_shape = conv_layer_2.out_shape;
    // Khởi tạo BatchNorm1d_Params
    BatchNorm1d_Params params3 = {NULL, NULL, NULL, NULL};
    // Đọc tham số từ các file (cho chế độ suy luận)
    BatchNorm1d_read_params(
        "C:/EDABK/TinyML/CNN_codeC/Model_inference/BatchNorm_Gammas.bin",
        "C:/EDABK/TinyML/CNN_codeC/Model_inference/BatchNorm_Betas.bin",
        "C:/EDABK/TinyML/CNN_codeC/Model_inference/BatchNorm_MovingMeans.bin",
        "C:/EDABK/TinyML/CNN_codeC/Model_inference/BatchNorm_MovingVariances.bin",
        &params3, input3_shape.channels
    );
    if (!params3.gamma || !params3.beta || !params3.moving_mean || !params3.moving_var) {
        printf("Error: Failed to load BatchNorm parameters\n");
        free(output2);
        return 1;
    }
    // Khởi tạo lớp BatchNorm1d
    BatchNorm1d BatchNorm1;
    BatchNorm1d_init(&BatchNorm1, input3_shape, params3, 0.001f, 0.99f, true, true);
    // Cấp phát bộ nhớ cho đầu ra
    float* output3 = (float*)calloc(input3_shape.batch_size * input3_shape.channels * input3_shape.height * input3_shape.width, sizeof(float));
    if (!output3) {
        printf("Error: Failed to allocate memory for output3\n");
        free(output2);
        BatchNorm1d_free(&BatchNorm1);
        return 1;
    }
    // Gọi lan truyền xuôi trong chế độ suy luận
    BatchNorm1d_forward(&BatchNorm1, output2, output3, false);
    // Lưu kết quả vào file
    char file_output3[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/batchnorm_output.bin";
    saveOutput_Conv2D_ToFile(file_output3, output3, input3_shape.batch_size, input3_shape.channels, input3_shape.height, input3_shape.width);
    
    printf("After BatchNorm: (%d, %d, %d, %d)\n", 
            BatchNorm1.out_shape.batch_size, 
            BatchNorm1.out_shape.channels, 
            BatchNorm1.out_shape.height, 
            BatchNorm1.out_shape.width);
    free(output2);
    BatchNorm1d_free(&BatchNorm1);
    //*************************************************************layer 4****************************************************************
    Feature_map_shape input4_shape = BatchNorm1.out_shape;
    Kernel_shape pool_size = {{2, 2}}; 
    // Parameters for Maxpool2d
    int padding_mp[] = {0, 0};
    int stride_mp[] = {2, 2};
    Layer maxpool1;

    float* output4 = (float*)malloc(input4_shape.batch_size * input4_shape.channels * 
        ((input4_shape.height + 2 * padding_mp[0] - pool_size.size[0]) / stride_mp[0] + 1) * 
        ((input4_shape.width + 2 * padding_mp[1] - pool_size.size[1]) / stride_mp[1] + 1) * sizeof(float));
    if (!output4) {
        printf("Error: Failed to allocate memory for output4\n");
        free(output3);
        return 1;
    }
    
    // Run maxpooling with conv output as input
    Maxpool2d(&maxpool1, input4_shape, pool_size, stride_mp, padding_mp, output3, output4);
    // Save maxpool output
    char file_output4[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/maxpool_output.bin";
    saveOutput_Conv2D_ToFile(file_output4, output4, maxpool1.out_shape.batch_size, maxpool1.out_shape.channels, maxpool1.out_shape.height, maxpool1.out_shape.width);
    
    printf("After maxpool2d: (%d, %d, %d, %d)\n", 
            maxpool1.out_shape.batch_size, 
            maxpool1.out_shape.channels, 
            maxpool1.out_shape.height, 
            maxpool1.out_shape.width);
    free(output3);
    //*************************************************************layer 5****************************************************************
    Feature_map_shape input5_shape = maxpool1.out_shape;
    float* output5 = (float*)malloc(input5_shape.batch_size * input5_shape.channels * input5_shape.height * input5_shape.width * sizeof(float));
    if (!output5) {
        printf("Error: Failed to allocate memory for output5\n");
        free(output4);
        return 1;
    }
    
    Flatten(output4, output5, input5_shape);

    char file_output5[] = "C:/EDABK/TinyML/CNN_codeC/Model_inference/flatten_output.bin";
    saveOutput_Conv2D_ToFile(file_output5, output5, input5_shape.batch_size, input5_shape.channels, input5_shape.height, input5_shape.width);

    printf("After Flatten: (%d, %d)\n", maxpool1.out_shape.batch_size, 
            input5_shape.channels * input5_shape.height * input5_shape.width);
    //*************************************************************layer 6****************************************************************
    int num_NodePreLayer5 = input5_shape.channels * input5_shape.height * input5_shape.width;
    int num_NodeThisLayer6 = 128;

    float* weights6 = NULL;
    int weight_size6;

    readData_FullConnect_FromFile("C:/EDABK/TinyML/CNN_codeC/Model_inference/dense1_weights.bin", &weights6, &weight_size6);
    if (!weights6) {
        printf("Error: Failed to load dense1 weights\n");
        free(output5);
        return 1;
    }
    
    Params dens1_prams = {weights6, NULL};
    float* output6 = (float*)malloc(batch_size * num_NodeThisLayer6 * sizeof(float));
    if (!output6) {
        printf("Error: Failed to allocate memory for output6\n");
        free(output5);
        free(weights6);
        return 1;
    }

    fullyConnected_forward(num_NodePreLayer5, num_NodeThisLayer6, batch_size, dens1_prams, false, output5, output6, RELU);
    
    saveOutput_FullConnect_ToFile("C:/EDABK/TinyML/CNN_codeC/Model_inference/dense1_output.bin", output6, batch_size, num_NodeThisLayer6);

    printf("After Dense 1: (%d, %d)\n", batch_size, num_NodeThisLayer6);
    free(output5);
    free(weights6);
    //*************************************************************layer 7****************************************************************
    int num_NodeThisLayer7 = 10;

    float* weights7 = NULL;
    int weight_size7;

    readData_FullConnect_FromFile("C:/EDABK/TinyML/CNN_codeC/Model_inference/dense2_weights.bin", &weights7, &weight_size7);
    if (!weights7) {
        printf("Error: Failed to load dense2 weights\n");
        free(output6);
        return 1;
    }
    
    Params dens2_prams = {weights7, NULL};
    float* output7 = (float*)malloc(batch_size * num_NodeThisLayer7 * sizeof(float));
    if (!output7) {
        printf("Error: Failed to allocate memory for output7\n");
        free(output6);
        free(weights7);
        return 1;
    }

    fullyConnected_forward(num_NodeThisLayer6, num_NodeThisLayer7, batch_size, dens2_prams, false, output6, output7, SOFTMAX);
    
    saveOutput_FullConnect_ToFile("C:/EDABK/TinyML/CNN_codeC/Model_inference/dense2_output.bin", output7, batch_size, num_NodeThisLayer7);

    printf("After Dense 2: (%d, %d)\n", batch_size, num_NodeThisLayer7);
    free(output6);
    free(weights7);
    //*******************************************predict**********************************************************
    int* pred_output = predict_labels(output7, batch_size, num_NodeThisLayer7);
    if (!pred_output) {
        printf("Error: Failed to predict labels\n");
        free(output7);
        return 1;
    }
    
    const char* true_output_file = "C:/EDABK/TinyML/CNN_codeC/Model_inference/true_output.bin";
    float accuracy = calculate_accuracy(pred_output, true_output_file, batch_size);
    printf("\nAccuracy: %.2f%%\n", accuracy);

    // Clean up
    free(output7);
    free(pred_output);
    // In thời gian toàn bộ chương trình
    clock_t end_program = clock();
    printf("Time execution of all program: %.3f minutes\n", 
           (double)(end_program - start_program) / CLOCKS_PER_SEC / 60.0);
    return 0;
}