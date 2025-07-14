#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#include <omp.h>

float relu(float x) {
    return (x > 0) ? x : 0;
}

// Hàm softmax
void softmax(float* input, int batch_size, int num_nodes, float* output) {
    for (int b = 0; b < batch_size; b++) {
        float* current_input = input + b * num_nodes;
        float* current_output = output + b * num_nodes;
        
        // Tính tổng lũy thừa e^z_i
        float sum_exp = 0.0;
        for (int i = 0; i < num_nodes; i++) {
            sum_exp += exp(current_input[i]);
        }
        
        // Tính softmax cho từng phần t     ử
        for (int i = 0; i < num_nodes; i++) {
            current_output[i] = exp(current_input[i]) / sum_exp;
        }
    }
}


//conv2d va maxpooling
typedef struct {
    int batch_size;
    int channels;
    int height;
    int width;
} Feature_map_shape;

typedef struct {
    int size[2];
} Kernel_shape;

typedef struct {
    float* weight;
    float* bias;
} Params;

typedef struct {
    int stride[2];
    int padding[2];
    bool use_bias;
    Feature_map_shape in_shape, out_shape;
    Kernel_shape kernel_sh;
    Params params;
} Layer;

// Forward function
void Conv2d_forward(Layer* conv, Feature_map_shape in_shape, int output_channels, Kernel_shape kernel_sh, Params params, int stride[2], int padding[2], bool use_bias, float* input, float* output) {
    if (!input || !output || !params.weight) {
        printf("Error: NULL pointer in Conv2d_forward\n");
        return;
    }
    conv->in_shape = in_shape;
    conv->out_shape.channels = output_channels;
    conv->kernel_sh = kernel_sh;
    conv->params = params;
    conv->stride[0] = stride[0]; 
    conv->stride[1] = stride[1]; 
    conv->padding[0] = padding[0];
    conv->padding[1] = padding[1];
    conv->use_bias = use_bias;
    conv->out_shape.height = (conv->in_shape.height + 2 * conv->padding[0] - conv->kernel_sh.size[0]) / conv->stride[0] + 1;     
    conv->out_shape.width = (conv->in_shape.width + 2 * conv->padding[1] - conv->kernel_sh.size[1]) / conv->stride[1] + 1;
    conv->out_shape.batch_size = conv->in_shape.batch_size; 

    const int   in_ch = in_shape.channels, 
                in_h = in_shape.height, 
                in_w = in_shape.width;
    const int   out_ch = output_channels, 
                out_h = conv->out_shape.height, 
                out_w = conv->out_shape.width;
    const int   k_h = kernel_sh.size[0], 
                k_w = kernel_sh.size[1];



    #pragma omp parallel for
    for (int b = 0; b < conv->in_shape.batch_size; b++) {
        int b_offset_in = b * in_ch * in_h * in_w;
        int b_offset_out = b * out_ch * out_h * out_w;
        for (int c = 0; c < out_ch; c++) {
            int c_offset_out = c * out_h * out_w;
            int c_offset_weight = c * in_ch * k_h * k_w;
            for (int h = 0; h < out_h; h++) {
                int h_offset_out = h * out_w;
                for (int w = 0; w < out_w; w++) {
                    float sum = 0;
                    for (int kh = 0; kh < k_h; kh++) {
                        int in_h_idx = h * stride[0] + kh - padding[0];
                        if (in_h_idx < 0 || in_h_idx >= in_h) continue;
                        int kh_offset_weight = kh * k_w;
                        for (int kw = 0; kw < k_w; kw++) {
                            int in_w_idx = w * stride[1] + kw - padding[1];
                            if (in_w_idx < 0 || in_w_idx >= in_w) continue;
                            for (int ic = 0; ic < in_ch; ic++) {
                                int ic_offset_in = ic * in_h * in_w;
                                int ic_offset_weight = ic * k_h * k_w;
                                int input_idx = b_offset_in + ic_offset_in + in_h_idx * in_w + in_w_idx;
                                int weight_idx = c_offset_weight + ic_offset_weight + kh_offset_weight + kw;
                                sum += input[input_idx] * params.weight[weight_idx];
                            }
                        }
                    }
                    int output_idx = b_offset_out + c_offset_out + h_offset_out + w;
                    output[output_idx] = relu(sum + (use_bias ? params.bias[c] : 0));
                }   
            }
        }
    }
}

// Maxpooling function
void Maxpool2d(Layer* maxpool, Feature_map_shape in_shape, Kernel_shape kernel_sh, int stride[2], int padding[2], float* input, float* output) {
    if (!input || !output) {
        printf("Error: NULL pointer in Maxpool2d\n");
        return;
    }
    maxpool->in_shape = in_shape;
    maxpool->kernel_sh = kernel_sh;
    maxpool->padding[0] = padding[0];
    maxpool->padding[1] = padding[1];
    maxpool->stride[0] = stride[0]; 
    maxpool->stride[1] = stride[1];

    int out_height = (maxpool->in_shape.height + 2 * maxpool->padding[0] - maxpool->kernel_sh.size[0]) / maxpool->stride[0] + 1;
    int out_width = (maxpool->in_shape.width + 2 * maxpool->padding[1] - maxpool->kernel_sh.size[1]) / maxpool->stride[1] + 1;
    
    maxpool->out_shape.batch_size = maxpool->in_shape.batch_size;
    maxpool->out_shape.height = out_height;     
    maxpool->out_shape.width = out_width;
    maxpool->out_shape.channels = in_shape.channels;

    const int   in_ch = in_shape.channels, 
                in_h = in_shape.height, 
                in_w = in_shape.width;
    const int   k_h = kernel_sh.size[0], 
                k_w = kernel_sh.size[1];

    for (int b = 0; b < in_shape.batch_size; b++) 
    {
        int b_offset = b * in_ch * in_h * in_w;
        int b_offset_out = b * in_ch * out_height * out_width;
        for (int ic = 0; ic < in_ch; ic++) 
        {
            int ic_offset = ic * in_h * in_w;
            int ic_offset_out = ic * out_height * out_width;
            for (int oh = 0; oh < out_height; oh++) 
            {
                int oh_offset = oh * out_width;
                for (int ow = 0; ow < out_width; ow++) 
                {
                    float max_val = -FLT_MAX;
                    for (int kh = 0; kh < k_h; kh++) 
                    {
                        int ih = oh * stride[0] + kh - padding[0];
                        if (ih < 0 || ih >= in_h) continue;
                        for (int kw = 0; kw < k_w; kw++) 
                        {
                            int iw = ow * stride[1] + kw - padding[1];
                            if (iw < 0 || iw >= in_w) continue;
                            int input_idx = b_offset + ic_offset + ih * in_w + iw;
                            float val = input[input_idx];
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    int output_idx = b_offset_out + ic_offset_out + oh_offset + ow;
                    output[output_idx] = max_val;
                }
            }
        }
    }
}

Feature_map_shape Layer_get_output_shape(const Layer* layer) { 
    return layer->out_shape; 
}

//fullyConnected_forward
typedef enum {
    NONE,
    RELU,
    SOFTMAX
} ActivationType;

//fullyConnected_forward
void fullyConnected_forward(int num_NodePreLayer, int num_NodeThisLayer, int batch_size, Params params, bool use_bias, float* input, float* output, ActivationType activation) {
    if (!input || !output || !params.weight) {
        printf("Error: NULL pointer in fullyConnected_forward\n");
        return;
    }
    for (int b = 0; b < batch_size; b++) {
        float* current_input = input + b * num_NodePreLayer;
        float* current_output = output + b * num_NodeThisLayer;
        
        // Tính  linear output
        for (int i = 0; i < num_NodeThisLayer; i++) {
            current_output[i] = use_bias ? params.bias[i] : 0;
            for (int j = 0; j < num_NodePreLayer; j++) {
                int weight_idx = i * num_NodePreLayer + j;
                current_output[i] += current_input[j] * params.weight[weight_idx];
            }
        }
        // Áp dụng activation
        if (activation == RELU) {
            for (int i = 0; i < num_NodeThisLayer; i++) {
                current_output[i] = relu(current_output[i]);
            }
        } else if (activation == SOFTMAX) {
            // Tính softmax cho batch hiện tại
            float sum_exp = 0.0;
            for (int i = 0; i < num_NodeThisLayer; i++) {
                sum_exp += exp(current_output[i]);
            }
        
            // Tính softmax cho từng phần tử
            for (int i = 0; i < num_NodeThisLayer; i++) {
                current_output[i] = exp(current_output[i]) / sum_exp;
            }
        } else if (activation == NONE) {
        // Không làm gì cả, giữ nguyên đầu ra
        }
    }
}

// void readData_Conv2D_FromFile(const char *filename, float** input) {
//     FILE *file = fopen(filename, "r");
//     if (file == NULL) {
//         printf("Failed to open file: %s\n", filename);
//         return;
//     }
    
//     int size = 0;
//     float value;
//     while (fscanf(file, "%f", &value) == 1) {
//         size++;
//     }
    
//     rewind(file);
    
//     *input = (float*)malloc(size * sizeof(float));
    
//     for (int i = 0; i < size; i++) {
//         if (fscanf(file, "%f", &(*input)[i]) != 1) {
//             break;
//         }
//     }
    
//     fclose(file);
// }

void readData_Conv2D_FromFile(const char *filename, float** input) {
    *input = NULL; // Khởi tạo để tránh truy cập con trỏ không hợp lệ
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Failed to open file: %s\n", filename);
        return;
    }
    
    int size = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        size++;
    }
    
    if (size == 0) {
        printf("Error: No valid float data found in file %s\n", filename);
        fclose(file);
        return;
    }

    rewind(file); 
    
    *input = (float*)malloc(size * sizeof(float));
    if (*input == NULL) {
        printf("Error: Memory allocation failed for %d floats\n", size);
        fclose(file);
        return;
    }
    
    for (int i = 0; i < size; i++) {
        if (fscanf(file, "%f", &(*input)[i]) != 1) {
            printf("Error: Failed to read data at index %d in file %s\n", i, filename);
            free(*input);
            *input = NULL;
            fclose(file);
            return;
        }
    }
    
    fclose(file);
}

// void saveOutput_Conv2D_ToFile(const char *filename, const float *output, int batch_size, int output_channels, int height, int width) {
//     FILE *file = fopen(filename, "w");
//     if (file == NULL) {
//         printf("Failed to open file for writing: %s\n", filename);
//         return;
//     }

//     int total_elements = batch_size * output_channels * height * width;

//     for (int i = 0; i < total_elements; i++) {
//         fprintf(file, "%.20f\n", output[i]);
//     }

//     fclose(file);
//     printf("Done writing %s\n", filename);
// }

void saveOutput_Conv2D_ToFile(const char *filename, const float *output, int batch_size, int output_channels, int height, int width) {
    if (output == NULL) {
        printf("Error: Output array is NULL\n");
        return;
    }
    if (batch_size <= 0 || output_channels <= 0 || height <= 0 || width <= 0) {
        printf("Error: Invalid dimensions: batch_size=%d, output_channels=%d, height=%d, width=%d\n",
               batch_size, output_channels, height, width);
        return;
    }

    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Failed to open file for writing: %s\n", filename);
        return;
    }

    int total_elements = batch_size * output_channels * height * width;

    for (int i = 0; i < total_elements; i++) {
        if (fprintf(file, "%.20f\n", output[i]) < 0) {
            printf("Error: Failed to write data at index %d to file %s\n", i, filename);
            fclose(file);
            return;
        }
    }

    fclose(file);
    printf("Done writing %s\n", filename);
}

// void readData_FullConnect_FromFile(const char* filename, float** input, int* size) {
//     FILE* file = fopen(filename, "r");
//     if (file == NULL) {
//         printf("Error: Could not open file %s for reading\n", filename);
//         *input = NULL;  // Set input to NULL to indicate failure
//         *size = 0;      // Set size to 0 to indicate no data
//         return;
//     }
    
//     float* temp = NULL;
//     int count = 0;
//     float value;
//     while (fscanf(file, "%f", &value) == 1) {
//         temp = (float*)realloc(temp, (count + 1) * sizeof(float));
//         temp[count++] = value;
//     }
    
//     fclose(file);
//     *input = temp;
//     *size = count;
// }

void readData_FullConnect_FromFile(const char* filename, float** input, int* size) {
    *input = NULL;  // Khởi tạo để tránh truy cập con trỏ không hợp lệ
    *size = 0;      // Khởi tạo size

    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error: Could not open file %s for reading\n", filename);
        return;
    }
    
    float* temp = NULL;
    int count = 0;
    float value;
    while (fscanf(file, "%f", &value) == 1) {
        float* new_temp = (float*)realloc(temp, (count + 1) * sizeof(float));
        if (new_temp == NULL) {
            printf("Error: Memory reallocation failed at index %d\n", count);
            free(temp);
            fclose(file);
            return;
        }
        temp = new_temp;
        temp[count++] = value;
    }
    
    if (count == 0) {
        printf("Error: No valid float data found in file %s\n", filename);
        fclose(file);
        return;
    }

    fclose(file);
    *input = temp;
    *size = count;
}

// void saveOutput_FullConnect_ToFile(const char* filename, float* output, int batch_size, int num_nodes) {
//     FILE* file = fopen(filename, "w");
//     if (file == NULL) {
//         printf("Error: Could not open file %s for writing\n", filename);
//         return;
//     }
    
//     int total_elements = batch_size * num_nodes;

//     for (int i = 0; i < total_elements; ++i) {
//         fprintf(file, "%.20e\n", output[i]);  // Định dạng kiểu khoa học (số mũ)
//     }
    
//     fclose(file);
//     printf("Done writing %s\n", filename);
// }

void saveOutput_FullConnect_ToFile(const char* filename, float* output, int batch_size, int num_nodes) {
    if (output == NULL) {
        printf("Error: Output array is NULL\n");
        return;
    }
    if (batch_size <= 0 || num_nodes <= 0) {
        printf("Error: Invalid dimensions: batch_size=%d, num_nodes=%d\n", batch_size, num_nodes);
        return;
    }

    FILE* file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }
    
    int total_elements = batch_size * num_nodes;

    for (int i = 0; i < total_elements; ++i) {
        if (fprintf(file, "%.20e\n", output[i]) < 0) {
            printf("Error: Failed to write data at index %d to file %s\n", i, filename);
            fclose(file);
            return;
        }
    }
    
    fclose(file);
    printf("Done writing %s\n", filename);
}

// reorder_BCHW_to_BHWC
void Flatten(const float *input, float *output, Feature_map_shape shape) {
    if (!input || !output) {
        printf("Error: NULL pointer in Flatten\n");
        return;
    }
    int batch_size = shape.batch_size;
    int channels = shape.channels;
    int height = shape.height;
    int width = shape.width;

    for (int b = 0; b < batch_size; b++) {  
        int b_offset_in = b * channels * height * width;
        int b_offset_out = b * height * width * channels;
        for (int h = 0; h < height; h++) {
            int h_offset_in = h * width;
            int h_offset_out = h * width * channels;
            for (int w = 0; w < width; w++) {
                int w_offset_out = w * channels;
                for (int c = 0; c < channels; c++) {
                    int index_input = b_offset_in + c * height * width + h_offset_in + w;
                    int index_output = b_offset_out + h_offset_out + w_offset_out + c;
                    output[index_output] = input[index_input];
                }
            }
        }
    }
}

int* predict_labels(float* output, int batch_size, int num_NodeThisLayer) {
    if (!output) {
        printf("Error: NULL output pointer in predict_labels\n");
        return NULL;
    }
    const char* class_names[] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };
    
    int* predictions = (int*)malloc(batch_size * sizeof(int));
    if (predictions == NULL) {
        printf("Error: Memory allocation failed in predict_labels\n");
        return NULL;
    }
    
    for (int i = 0; i < batch_size; i++) {
        float max = output[i * num_NodeThisLayer];
        int predict = 0;
        
        for (int j = 0; j < num_NodeThisLayer; j++) {
            int current_idx = i * num_NodeThisLayer + j;
            if (max < output[current_idx]) {
                max = output[current_idx];
                predict = j;
            }
        }
        predictions[i] = predict;
        printf("Sample %d - Predicted class: %s (index: %d)\n", i, class_names[predict], predict);
    }
    
    return predictions;
}

float calculate_accuracy(int* pred_output, const char* true_output_file, int num_samples) {
    if (!pred_output) {
        printf("Error: NULL pred_output pointer in calculate_accuracy\n");
        return -1.0f;
    }
    FILE* file = fopen(true_output_file, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", true_output_file);
        return -1.0f;
    }

    int correct_predictions = 0;
    int true_value;
    for (int i = 0; i < num_samples; i++) {
        if (fscanf(file, "%d", &true_value) == 1) {
            if (pred_output[i] == true_value) {
                correct_predictions++;
            }
        }
        fgetc(file);
    }

    fclose(file);
    return (float)correct_predictions / num_samples * 100.0f;
}

// Cấu trúc cho các tham số của BatchNorm1d
typedef struct {
    float* gamma;  // Tham số tỷ lệ
    float* beta;   // Tham số dịch chuyển
    float* moving_mean;   // Trung bình chạy moving_mean
    float* moving_var;    // Phương sai chạy moving_var
} BatchNorm1d_Params;

// Cấu trúc cho lớp BatchNorm1d
typedef struct {
    Feature_map_shape in_shape;   // Hình dạng đầu vào
    Feature_map_shape out_shape;  // Hình dạng đầu ra
    BatchNorm1d_Params params;    // Tham số của lớp
    float eps;                    // Hằng số nhỏ để ổn định số học
    float momentum;               // Động lượng cho cập nhật trung bình/phương sai chạy
    bool center;                  // Có dịch chuyển bằng beta hay không
    bool scale;                   // Có điều chỉnh tỷ lệ bằng gamma hay không
} BatchNorm1d;

// Khởi tạo tham số cho chế độ huấn luyện
// void BatchNorm1d_init_params_training(BatchNorm1d_Params* params, int channels, float beta_init, float gamma_init, float moving_mean_init, float moving_var_init) {
//     params->beta = (float*)calloc(channels, sizeof(float));
//     params->gamma = (float*)calloc(channels, sizeof(float));
//     params->moving_mean = (float*)calloc(channels, sizeof(float));
//     params->moving_var = (float*)calloc(channels, sizeof(float));

//     if (!params->beta || !params->gamma || !params->moving_mean || !params->moving_var) {
//         printf("Error: Unable to allocate memory for BatchNorm1d parameters\n");
//         free(params->beta);
//         free(params->gamma);
//         free(params->moving_mean);
//         free(params->moving_var);
//         params->beta = params->gamma = params->moving_mean = params->moving_var = NULL;
//         return;
//     }

//     for (int c = 0; c < channels; c++) {
//         params->beta[c] = beta_init;
//         params->gamma[c] = gamma_init;
//         params->moving_mean[c] = moving_mean_init;
//         params->moving_var[c] = moving_var_init;
//     }
// }

// Khởi tạo lớp BatchNorm1d
void BatchNorm1d_init(BatchNorm1d* layer, Feature_map_shape in_shape, BatchNorm1d_Params params, float eps, float momentum, bool center, bool scale) {
    layer->in_shape = in_shape;
    layer->out_shape = in_shape; // Hình dạng đầu ra giống đầu vào
    layer->params = params;
    layer->eps = eps;
    layer->momentum = momentum;
    layer->center = center;
    layer->scale = scale;
}

// Lan truyền xuôi cho BatchNorm1d
void BatchNorm1d_forward(BatchNorm1d* layer, float* input, float* output, bool training) {
    int batch_size = layer->in_shape.batch_size;  
    int channels = layer->in_shape.channels;     
    int height = layer->in_shape.height;         
    int width = layer->in_shape.width;            
    int spatial_dim = height * width;             // Kích thước không gian (chiều cao * chiều rộng)

    // Cấp phát mảng tạm cho trung bình và phương sai của lô
    float* batch_mean = (float*)calloc(channels, sizeof(float));
    float* batch_var = (float*)calloc(channels, sizeof(float));
    if (!batch_mean || !batch_var) {
        printf("Error: Unable to allocate memory in BatchNorm1d_forward\n");
        free(batch_mean);
        free(batch_var);
        return;
    }

    if (training) {
        // Tính trung bình và phương sai cho mỗi kênh
        for (int c = 0; c < channels; c++) {
            float mean = 0.0f;
            for (int b = 0; b < batch_size; b++) 
            {
                for (int h = 0; h < height; h++) 
                {
                    for (int w = 0; w < width; w++) 
                    {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        mean += input[idx];
                    }
                }
            }
            mean /= (float)(batch_size * spatial_dim);
            batch_mean[c] = mean;

            float var = 0.0f;
            for (int b = 0; b < batch_size; b++) 
            {
                for (int h = 0; h < height; h++) 
                {
                    for (int w = 0; w < width; w++) 
                    {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        var += (input[idx] - mean) * (input[idx] - mean);
                    }
                }
            }
            var /= (float)(batch_size * spatial_dim);
            batch_var[c] = var;


            // running_mean = momentum * running_mean + (1 - momentum) * batch_mean;
            // running_var  = momentum * running_var  + (1 - momentum) * batch_var;

            // Cập nhật trung bình và phương sai chạy dùng cho inference
            layer->params.moving_mean[c] = layer->momentum * layer->params.moving_mean[c] + (1.0f - layer->momentum) * mean;
            layer->params.moving_var[c] = layer->momentum * layer->params.moving_var[c] + (1.0f - layer->momentum) * var;
        }

        // Chuẩn hóa và điều chỉnh tỷ lệ/dịch chuyển
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        float std = sqrtf(batch_var[c] + layer->eps);
                        output[idx] = (input[idx] - batch_mean[c]) / std;
                        if (layer->scale) {
                            output[idx] *= layer->params.gamma[c];
                        }
                        if (layer->center) {
                            output[idx] += layer->params.beta[c];
                        }
                    }
                }
            }
        }
    } 
    else {
        // Chế độ suy luận: sử dụng trung bình và phương sai chạy
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        float std = sqrtf(layer->params.moving_var[c] + layer->eps);
                        output[idx] = (input[idx] - layer->params.moving_mean[c]) / std;
                        if (layer->scale) {
                            output[idx] *= layer->params.gamma[c];
                        }
                        if (layer->center) {
                            output[idx] += layer->params.beta[c];
                        }
                    }
                }
            }
        }
    }

    free(batch_mean);
    free(batch_var);
}

// Giải phóng tham số của BatchNorm1d
void BatchNorm1d_free(BatchNorm1d* layer) {
    if (layer->params.gamma) free(layer->params.gamma);
    if (layer->params.beta) free(layer->params.beta);
    if (layer->params.moving_mean) free(layer->params.moving_mean);
    if (layer->params.moving_var) free(layer->params.moving_var);
}

//Đọc tham số của BatchNorm1d từ file
void BatchNorm1d_read_params(const char* gamma_file, const char* beta_file, 
                             const char* mean_file, const char* var_file, 
                             BatchNorm1d_Params* params, int channels) {
    readData_Conv2D_FromFile(gamma_file, &params->gamma);
    readData_Conv2D_FromFile(beta_file, &params->beta);
    readData_Conv2D_FromFile(mean_file, &params->moving_mean);
    readData_Conv2D_FromFile(var_file, &params->moving_var);
}