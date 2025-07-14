#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "CNN_model.h"
#include <time.h>

int main() {
    // Đo thời gian toàn bộ chương trình
    clock_t start_program = clock();

    // Tổng số ảnh và kích thước mỗi batch
    int total_images = 10000;
    int batch_size = 500; // Kích thước mỗi batch nhỏ
    int num_batches = (total_images + batch_size - 1) / batch_size; // Số batch cần xử lý

    // Tính số phần tử của mỗi ảnh
    int channels = 3, height = 32, width = 32;
    int elements_per_image = channels * height * width; // 3072

    // Đọc toàn bộ dữ liệu đầu vào từ file input.bin
    float* input = NULL;
    const char* file_input = "E:/EDABK/Project_1/CNN_code/CNN_model/input.bin";
    readData_Conv2D_FromFile(file_input, &input);

    if (input == NULL) {
        printf("Error: Failed to read input data from file %s\n", file_input);
        return 1;
    }
    
    // Khởi tạo mảng để lưu toàn bộ kết quả dự đoán
    int* all_predictions = (int*)malloc(total_images * sizeof(int));
    if (all_predictions == NULL) {
        printf("Error: Failed to allocate memory for all_predictions\n");
        free(input);
        return 1;
    }

    // Vòng lặp để xử lý từng batch
    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        // Xác định kích thước batch hiện tại
        int current_batch_size = (batch_idx + 1 == num_batches) ? (total_images - batch_idx * batch_size) : batch_size;
        bool use_bias = false;
        //*************************************************************layer 1****************************************************************
        Feature_map_shape input_shape = {current_batch_size, 3, 32, 32};
        Kernel_shape kernel_shape = {{3, 3}}; 
        Params params1 = {NULL, NULL};
        
        // Parameters for Conv2d
        int output_channels_1 = 32;
        int padding[] = {1, 1};
        int stride[] = {1, 1};

        Layer conv_layer_1;
        // Load weight 1
        char conv1_weight[] = "E:/EDABK/Project_1/CNN_code/CNN_model/conv1_weights.bin";
        readData_Conv2D_FromFile(conv1_weight, &params1.weight);
        if (params1.weight == NULL) {
            printf("Error: Failed to load conv1 weights\n");
            free(input);
            free(all_predictions);
            return 1;
        }
        
        // Allocate memory for batch input and output for conv layer
        float* batch_input = (float*)malloc(current_batch_size * input_shape.channels * input_shape.height * input_shape.width * sizeof(float));
        if (batch_input == NULL) {
            printf("Error: Failed to allocate memory for batch_input\n");
            free(params1.weight);
            free(input);
            free(all_predictions);
            return 1;
        }

        // Sao chép dữ liệu của batch hiện tại từ input vào batch_input
        int start_idx = batch_idx * batch_size * elements_per_image;
        for (int i = 0; i < current_batch_size * elements_per_image; i++) {
            batch_input[i] = input[start_idx + i];
        }

        float* output1 = (float*)malloc(input_shape.batch_size * output_channels_1 * input_shape.height * input_shape.width * sizeof(float));
        if (output1 == NULL) {
            printf("Error: Failed to allocate memory for output1\n");
            free(batch_input);
            free(params1.weight);
            free(input);
            free(all_predictions);
            return 1;
        }
        
        // Run forward pass for conv layer
        Conv2d_forward(&conv_layer_1, input_shape, output_channels_1, kernel_shape, params1, stride, padding, use_bias, batch_input, output1);

        // Print output shapes
        printf("Batch %d - After conv2d_1: (%d, %d, %d, %d)\n", 
               batch_idx, conv_layer_1.out_shape.batch_size, 
               conv_layer_1.out_shape.channels, 
               conv_layer_1.out_shape.height, 
               conv_layer_1.out_shape.width);
        free(batch_input);
        free(params1.weight);
        //*************************************************************layer 2****************************************************************
        Feature_map_shape input2_shape = conv_layer_1.out_shape;
        Params params2 = {NULL, NULL};
        int output_channels_2 = 64;
        Layer conv_layer_2;
        
        float* output2 = (float*)malloc(input2_shape.batch_size * output_channels_2 * input2_shape.height * input2_shape.width * sizeof(float));
        if (output2 == NULL) {
            printf("Error: Failed to allocate memory for output2\n");
            free(output1);
            free(input);
            free(all_predictions);
            return 1;
        }
        // Load weight 2
        char conv2_weight[] = "E:/EDABK/Project_1/CNN_code/CNN_model/conv2_weights.bin";
        readData_Conv2D_FromFile(conv2_weight, &params2.weight);
        if (params2.weight == NULL) {
            printf("Error: Failed to load conv2 weights\n");
            free(output2);
            free(output1);
            free(input);
            free(all_predictions);
            return 1;
        }
        // Run forward pass for conv layer
        Conv2d_forward(&conv_layer_2, input2_shape, output_channels_2, kernel_shape, params2, stride, padding, use_bias, output1, output2);
        // Print output shapes
        printf("Batch %d - After conv2d_2: (%d, %d, %d, %d)\n", 
               batch_idx, conv_layer_2.out_shape.batch_size, 
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
            "E:/EDABK/Project_1/CNN_code/CNN_model/BatchNorm_Gammas.bin",
            "E:/EDABK/Project_1/CNN_code/CNN_model/BatchNorm_Betas.bin",
            "E:/EDABK/Project_1/CNN_code/CNN_model/BatchNorm_MovingMeans.bin",
            "E:/EDABK/Project_1/CNN_code/CNN_model/BatchNorm_MovingVariances.bin",
            &params3, input3_shape.channels
        );
        if (!params3.gamma || !params3.beta || !params3.moving_mean || !params3.moving_var) {
            printf("Error: Failed to load BatchNorm parameters\n");
            free(output2);
            free(input);
            free(all_predictions);
            return 1;
        }
        // Khởi tạo lớp BatchNorm1d
        BatchNorm1d BatchNorm1;
        BatchNorm1d_init(&BatchNorm1, input3_shape, params3, 0.001f, 0.99f, true, true);
        // Cấp phát bộ nhớ cho đầu ra
        float* output3 = (float*)calloc(input3_shape.batch_size * input3_shape.channels * input3_shape.height * input3_shape.width, sizeof(float));
        if (output3 == NULL) {
            printf("Error: Failed to allocate memory for output3\n");
            free(output2);
            BatchNorm1d_free(&BatchNorm1);
            free(input);
            free(all_predictions);
            return 1;
        }
        // Gọi lan truyền xuôi trong chế độ suy luận
        BatchNorm1d_forward(&BatchNorm1, output2, output3, false);
        
        printf("Batch %d - After BatchNorm: (%d, %d, %d, %d)\n", 
               batch_idx, BatchNorm1.out_shape.batch_size, 
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
        if (output4 == NULL) {
            printf("Error: Failed to allocate memory for output4\n");
            free(output3);
            free(input);
            free(all_predictions);
            return 1;
        }
        
        // Run maxpooling with batchnorm output as input
        Maxpool2d(&maxpool1, input4_shape, pool_size, stride_mp, padding_mp, output3, output4);
        
        printf("Batch %d - After maxpool2d: (%d, %d, %d, %d)\n", 
               batch_idx, maxpool1.out_shape.batch_size, 
               maxpool1.out_shape.channels, 
               maxpool1.out_shape.height, 
               maxpool1.out_shape.width);
        free(output3);
        //*************************************************************layer 5****************************************************************
        Feature_map_shape input5_shape = maxpool1.out_shape;
        float* output5 = (float*)malloc(input5_shape.batch_size * input5_shape.channels * input5_shape.height * input5_shape.width * sizeof(float));
        if (output5 == NULL) {
            printf("Error: Failed to allocate memory for output5\n");
            free(output4);
            free(input);
            free(all_predictions);
            return 1;
        }
        
        Flatten(output4, output5, input5_shape);

        printf("Batch %d - After Flatten: (%d, %d)\n", batch_idx, maxpool1.out_shape.batch_size, 
                input5_shape.channels * input5_shape.height * input5_shape.width);
        free(output4);
        //*************************************************************layer 6****************************************************************
        int num_NodePreLayer5 = input5_shape.channels * input5_shape.height * input5_shape.width;
        int num_NodeThisLayer6 = 128;

        float* weights6 = NULL;
        int weight_size6;

        readData_FullConnect_FromFile("E:/EDABK/Project_1/CNN_code/CNN_model/dense1_weights.bin", &weights6, &weight_size6);
        if (weights6 == NULL) {
            printf("Error: Failed to load dense1 weights\n");
            free(output5);
            free(input);
            free(all_predictions);
            return 1;
        }
        
        Params dens1_prams = {weights6, NULL};
        float* output6 = (float*)malloc(current_batch_size * num_NodeThisLayer6 * sizeof(float));
        if (output6 == NULL) {
            printf("Error: Failed to allocate memory for output6\n");
            free(weights6);
            free(output5);
            free(input);
            free(all_predictions);
            return 1;
        }

        fullyConnected_forward(num_NodePreLayer5, num_NodeThisLayer6, current_batch_size, dens1_prams, false, output5, output6, RELU);
        
        printf("Batch %d - After Dense 1: (%d, %d)\n", batch_idx, current_batch_size, num_NodeThisLayer6);
        free(output5);
        free(weights6);
        //*************************************************************layer 7****************************************************************
        int num_NodeThisLayer7 = 10;

        float* weights7 = NULL;
        int weight_size7;

        readData_FullConnect_FromFile("E:/EDABK/Project_1/CNN_code/CNN_model/dense2_weights.bin", &weights7, &weight_size7);
        if (weights7 == NULL) {
            printf("Error: Failed to load dense2 weights\n");
            free(output6);
            free(input);
            free(all_predictions);
            return 1;
        }
        
        Params dens2_prams = {weights7, NULL};
        float* output7 = (float*)malloc(current_batch_size * num_NodeThisLayer7 * sizeof(float));
        if (output7 == NULL) {
            printf("Error: Failed to allocate memory for output7\n");
            free(weights7);
            free(output6);
            free(input);
            free(all_predictions);
            return 1;
        }

        fullyConnected_forward(num_NodeThisLayer6, num_NodeThisLayer7, current_batch_size, dens2_prams, false, output6, output7, SOFTMAX);
        
        printf("Batch %d - After Dense 2: (%d, %d)\n", batch_idx, current_batch_size, num_NodeThisLayer7);
        free(output6);
        free(weights7);
        //*******************************************predict**********************************************************
        int* pred_output = predict_labels(output7, current_batch_size, num_NodeThisLayer7);
        if (pred_output == NULL) {
            printf("Error: Prediction failed for batch %d\n", batch_idx);
            free(output7);
            free(input);
            free(all_predictions);
            return 1;
        }

        // Lưu kết quả dự đoán vào mảng tổng
        for (int i = 0; i < current_batch_size; i++) {
            all_predictions[batch_idx * batch_size + i] = pred_output[i];
        }
        free(pred_output);
        free(output7);
    }

    free(input); // Giải phóng bộ nhớ của input sau khi sử dụng xong
 
    const char* true_output_file = "E:/EDABK/Project_1/CNN_code/CNN_model/true_output.bin";
    // Tính độ chính xác trên toàn bộ dữ liệu
    float accuracy = calculate_accuracy(all_predictions, true_output_file, total_images);
    if (accuracy < 0) {
        printf("Error: Failed to calculate accuracy\n");
        free(all_predictions);
        return 1;
    }
    printf("\nAccuracy: %.2f%%\n", accuracy);

    // Clean up
    free(all_predictions);

    // In thời gian toàn bộ chương trình
    clock_t end_program = clock();
    printf("Time execution of all program: %.3f seconds\n", 
           (double)(end_program - start_program) / CLOCKS_PER_SEC);
    return 0;
}
