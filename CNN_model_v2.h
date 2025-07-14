#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>

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

float loss_sparse_categorical_crossentropy(int num_classes, int batch_size, float* y_pred, int* y_true) {
    float loss = 0;
    for(int b = 0; b < batch_size; b++) {
        float* current_y_pred = y_pred + b * num_classes;
        int true_label = y_true[b];
        float prob = current_y_pred[true_label];
        // Kẹp giá trị prob để tránh log(0)
        if (prob < 1e-7) prob = 1e-7;
        if (prob > 1 - 1e-7) prob = 1 - 1e-7;
        loss -= logf(prob);
    }
    return loss / batch_size;
}

void backward_sparse_categorical_crossentropy_with_softmax(int num_classes, int batch_size, float* input, int* y_true, float* grad_input) {
    for (int i = 0; i < batch_size * num_classes; i++) {
        grad_input[i] = 0.0f;
    }

    float* softmax_probs = (float*)malloc(batch_size * num_classes * sizeof(float));
    if (!softmax_probs) {
        printf("Error: Memory allocation failed for softmax_probs\n");
        return;
    }

    softmax(input, batch_size, num_classes, softmax_probs);

    // Tính gradient cho mỗi mẫu
    // dL/dinput[b, i] = (1/batch_size)*(softmax_probs[i] - δ[i,true_label])
    // δ[i,true_label = 1 nếu i==true_label, và 0 nếu không.
    for (int b = 0; b < batch_size; b++) {
        float* current_grad_input = grad_input + b * num_classes;
        float* current_softmax_probs = softmax_probs + b * num_classes;
        int true_label = y_true[b];

        // Tính gradient: (softmax_probs[i] - delta[i, true_label]) / batch_size
        for (int i = 0; i < num_classes; i++) {
            float delta = (i == true_label) ? 1.0f : 0.0f;
            current_grad_input[i] = (current_softmax_probs[i] - delta) / (float)batch_size;
        }
    }
    free(softmax_probs);
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
    Feature_map_shape in_shape, out_shape;
    Kernel_shape kernel_sh;
    Params params;
    int stride[2];
    int padding[2];
    bool use_bias;
    int* max_indexList;
} Layer;

typedef enum {
    NONE,
    RELU,
    SOFTMAX
} ActivationType;

typedef struct {
    float* gamma;
    float* beta;
    float* moving_mean;
    float* moving_var;
} BatchNorm1d_Params;

typedef struct {
    float* batch_mean;  // Trung bình lô
    float* batch_var;   // Phương sai lô
} BatchNorm1d_Stats;

typedef struct {
    Feature_map_shape in_shape;
    Feature_map_shape out_shape;
    BatchNorm1d_Params params;
    BatchNorm1d_Stats stats;
    float eps;
    float momentum;
    bool center;
    bool scale;
} BatchNorm1d;

void BatchNorm1d_init(BatchNorm1d* layer, Feature_map_shape in_shape, 
                    BatchNorm1d_Params params, float eps, float momentum,
                    bool center, bool scale) 
{
    layer->in_shape = in_shape;
    layer->out_shape = in_shape;
    layer->params = params;
    layer->eps = eps;
    layer->momentum = momentum;
    layer->center = center;
    layer->scale = scale;
    layer->stats.batch_mean = NULL;
    layer->stats.batch_var = NULL;
}

void BatchNorm1d_forward(BatchNorm1d* layer, float* input, float* output, bool training) 
{
    int batch_size = layer->in_shape.batch_size;
    int channels = layer->in_shape.channels;
    int height = layer->in_shape.height;
    int width = layer->in_shape.width;
    int spatial_dim = height * width;
    int N = batch_size * spatial_dim;

    // Giải phóng thống kê lô cũ nếu có
    if (layer->stats.batch_mean) free(layer->stats.batch_mean);
    if (layer->stats.batch_var) free(layer->stats.batch_var);
    layer->stats.batch_mean = NULL;
    layer->stats.batch_var = NULL;

    float* batch_mean = (float*)calloc(channels, sizeof(float));
    float* batch_var = (float*)calloc(channels, sizeof(float));

    if (!batch_mean || !batch_var) {
        printf("Error: Unable to allocate memory in BatchNorm1d_forward\n");
        free(batch_mean);
        free(batch_var);
        return;
    }

    if (training) {
        // Tính batch_mean
        for (int c = 0; c < channels; c++) {
            double sum = 0.0;
            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        sum += input[idx];
                    }
                }
            }
            batch_mean[c] = (float)(sum / N);
        }

        // Tính batch_var (sử dụng N-1 để khớp với TensorFlow)
        for (int c = 0; c < channels; c++) {
            double sum_squares = 0.0;
            for (int b = 0; b < batch_size; b++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        double x_minus_mean = input[idx] - batch_mean[c];
                        sum_squares += x_minus_mean * x_minus_mean;
                    }
                }
            }
            batch_var[c] = (float)(sum_squares / N); // Unbiased variance
        }

        // Lưu thống kê lô
        layer->stats.batch_mean = batch_mean;
        layer->stats.batch_var = batch_var;

        // Chuẩn hóa và áp dụng gamma, beta
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                double inv_std = 1.0 / sqrt((double)batch_var[c] + layer->eps);
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        double x_minus_mean = input[idx] - batch_mean[c];
                        double x_hat = x_minus_mean * inv_std;
                        output[idx] = (float)((layer->scale ? layer->params.gamma[c] : 1.0f) * x_hat + 
                                             (layer->center ? layer->params.beta[c] : 0.0f));
                    }
                }
            }
        }

        // Cập nhật moving_mean và moving_var
        for (int c = 0; c < channels; c++) {
            layer->params.moving_mean[c] = layer->momentum * layer->params.moving_mean[c] + 
                                          (1.0f - layer->momentum) * batch_mean[c];
            layer->params.moving_var[c] = layer->momentum * layer->params.moving_var[c] + 
                                         (1.0f - layer->momentum) * batch_var[c];
        }
    } else {
        // Chế độ suy luận
        for (int b = 0; b < batch_size; b++) {
            for (int c = 0; c < channels; c++) {
                double inv_std = 1.0 / sqrt((double)layer->params.moving_var[c] + layer->eps);
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        double x_hat = (input[idx] - layer->params.moving_mean[c]) * inv_std;
                        output[idx] = (float)((layer->scale ? layer->params.gamma[c] : 1.0f) * x_hat + 
                                             (layer->center ? layer->params.beta[c] : 0.0f));
                    }
                }
            }
        }
        free(batch_mean);
        free(batch_var);
    }
}

void BatchNorm1d_backward(BatchNorm1d* layer, float* forward_input, float* dout, 
    float* dinput, float learning_rate) 
{
    int batch_size = layer->in_shape.batch_size;
    int channels = layer->in_shape.channels;
    int height = layer->in_shape.height;
    int width = layer->in_shape.width;
    int spatial_dim = height * width;
    int N = batch_size * spatial_dim;

    float* batch_mean = layer->stats.batch_mean;
    float* batch_var = layer->stats.batch_var;
    if (!batch_mean || !batch_var) {
        printf("Error: Batch statistics not available in BatchNorm1d_backward\n");
        return;
    }

    // Cấp phát bộ nhớ cho các gradient tạm thời
    float* dgamma = (float*)calloc(channels, sizeof(float));
    float* dbeta = (float*)calloc(channels, sizeof(float));
    float* d_xhat = (float*)calloc(batch_size * channels * spatial_dim, sizeof(float));
    float* d_variance = (float*)calloc(channels, sizeof(float));
    float* d_mean = (float*)calloc(channels, sizeof(float));

    if (!dgamma || !dbeta || !d_xhat || !d_variance || !d_mean) {
        printf("Error: Unable to allocate memory in BatchNorm1d_backward\n");
        free(dgamma);
        free(dbeta);
        free(d_xhat);
        free(d_variance);
        free(d_mean);
        return;
    }

   // Tính d_xhat, dgamma, dbeta
   for (int c = 0; c < channels; c++) {
        double inv_std = 1.0 / sqrt((double)batch_var[c] + layer->eps);
        float gamma = layer->scale ? layer->params.gamma[c] : 1.0f;

        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                    double x_minus_mean = forward_input[idx] - batch_mean[c];
                    double x_hat = x_minus_mean * inv_std;
                    if (layer->scale) {
                        dgamma[c] += dout[idx] * (float)x_hat;
                    }
                    if (layer->center) {
                        dbeta[c] += dout[idx];
                    }
                    d_xhat[idx] = dout[idx] * gamma;
                }
            }
        }
    }

    // Cập nhật gamma và beta
    for (int c = 0; c < channels; c++) {
        if (layer->scale) {
            layer->params.gamma[c] -= learning_rate * dgamma[c];
        }
        if (layer->center) {
            layer->params.beta[c] -= learning_rate * dbeta[c];
        }
    }

    // Tính d_variance và d_mean
    for (int c = 0; c < channels; c++) {
        double inv_std = 1.0 / sqrt((double)batch_var[c] + layer->eps);
        double inv_std_cubed = inv_std * inv_std * inv_std;
        double sum_d_xhat = 0.0;
        double sum_d_xhat_times_x = 0.0;
        double sum_x_minus_mean = 0.0;

        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                    double x_minus_mean = forward_input[idx] - batch_mean[c];
                    sum_d_xhat += d_xhat[idx];
                    sum_d_xhat_times_x += d_xhat[idx] * x_minus_mean;
                    sum_x_minus_mean += x_minus_mean;
                }
            }
        }
        d_variance[c] = (float)(-0.5 * sum_d_xhat_times_x * inv_std_cubed);
        d_mean[c] = (float)(-sum_d_xhat * inv_std + d_variance[c] * (-2.0 / N) * sum_x_minus_mean);
    }

    // Tính dinput
    for (int c = 0; c < channels; c++) {
        double inv_std = 1.0 / sqrt((double)batch_var[c] + layer->eps);
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                        int idx = b * channels * spatial_dim + c * spatial_dim + h * width + w;
                        double x_minus_mean = forward_input[idx] - batch_mean[c];
                        dinput[idx] = (float)(d_xhat[idx] * inv_std + 
                        d_variance[c] * (2.0 / N) * x_minus_mean + 
                        d_mean[c] * (1.0 / N));
                }
            }
        }
    }

    // Giải phóng bộ nhớ
    free(dgamma);
    free(dbeta);
    free(d_xhat);
    free(d_variance);
    free(d_mean);
}

void BatchNorm1d_free(BatchNorm1d* layer) {
    if (layer->stats.batch_mean) free(layer->stats.batch_mean);
    if (layer->stats.batch_var) free(layer->stats.batch_var);
}

int Layer_get_size_weight(Layer* layer) { 
    return layer->out_shape.channels * layer->in_shape.channels * layer->kernel_sh.size[0] * layer->kernel_sh.size[1]; 
}


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
    for (int b = 0; b < conv->in_shape.batch_size; b++) 
    {
        int b_offset_in = b * in_ch * in_h * in_w;
        int b_offset_out = b * out_ch * out_h * out_w;
        for (int c = 0; c < out_ch; c++) 
        {
            int c_offset_out = c * out_h * out_w;
            int c_offset_weight = c * in_ch * k_h * k_w;
            for (int h = 0; h < out_h; h++) 
            {
                int h_offset_out = h * out_w;
                for (int w = 0; w < out_w; w++) {
                    float sum = 0;
                    for (int ic = 0; ic < in_ch; ic++) 
                    {
                        int ic_offset_in = ic * in_h * in_w;
                        int ic_offset_weight = ic * k_h * k_w;
                        for (int kh = 0; kh < k_h; kh++) 
                        {
                            int in_h_idx = h * stride[0] + kh - padding[0];
                            if (in_h_idx < 0 || in_h_idx >= in_h) continue;
                            int kh_offset_weight = kh * k_w;
                            for (int kw = 0; kw < k_w; kw++) 
                            {
                                int in_w_idx = w * stride[1] + kw - padding[1];
                                if (in_w_idx < 0 || in_w_idx >= in_w) continue;
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

// Backward function
void Conv2d_backward(Layer* conv, float *output_grad, float *input_grad, float learning_rate, float *input) 
{
    int weight_size = Layer_get_size_weight(conv);
    float* kernel_grad = (float*)calloc(weight_size, sizeof(float));

    int input_total_size = conv->in_shape.batch_size * conv->in_shape.channels * conv->in_shape.height * conv->in_shape.width;
    int output_total_size = conv->out_shape.batch_size * conv->out_shape.channels * conv->out_shape.height * conv->out_shape.width;

    // Reset input gradient
    for (int i = 0; i < input_total_size; i++) {
        input_grad[i] = 0.0f;
    }

    // Calculate input, kernel gradient
    #pragma omp parallel for
    for (int b = 0; b < conv->in_shape.batch_size; b++) {
        for (int c = 0; c < conv->out_shape.channels; c++) {
            for (int h = 0; h < conv->out_shape.height; h++) {
                for (int w = 0; w < conv->out_shape.width; w++) {
                    int output_idx = b * conv->out_shape.channels * conv->out_shape.height * conv->out_shape.width +
                                     c * conv->out_shape.height * conv->out_shape.width +
                                     h * conv->out_shape.width + w;

                    float grad_out = output_grad[output_idx];

                    for (int ic = 0; ic < conv->in_shape.channels; ic++) {
                        for (int kh = 0; kh < conv->kernel_sh.size[0]; kh++) {
                            for (int kw = 0; kw < conv->kernel_sh.size[1]; kw++) {
                                int ih = h * conv->stride[0] + kh - conv->padding[0];
                                int iw = w * conv->stride[1] + kw - conv->padding[1];

                                if (ih >= 0 && ih < conv->in_shape.height &&
                                    iw >= 0 && iw < conv->in_shape.width) {
                                    
                                    int input_idx = b * conv->in_shape.channels * conv->in_shape.height * conv->in_shape.width +
                                                    ic * conv->in_shape.height * conv->in_shape.width +
                                                    ih * conv->in_shape.width + iw;

                                    int weight_idx = c * conv->in_shape.channels * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] +
                                                     ic * conv->kernel_sh.size[0] * conv->kernel_sh.size[1] +
                                                     kh * conv->kernel_sh.size[1] + kw;

                                    input_grad[input_idx] += conv->params.weight[weight_idx] * grad_out;
                                    kernel_grad[weight_idx] += input[input_idx] * grad_out;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Update weights
    for (int i = 0; i < weight_size; i++) {
        conv->params.weight[i] -= learning_rate * kernel_grad[i];
    }

    free(kernel_grad);
}


// Maxpooling forward function
void Maxpool_forward(Layer* maxpool, Feature_map_shape in_shape, Kernel_shape kernel_sh, int stride[2], int padding[2], float* input, float* output) {
    maxpool->in_shape = in_shape;
    maxpool->kernel_sh = kernel_sh;
    maxpool->padding[0] = padding[0];
    maxpool->padding[1] = padding[1];
    maxpool->stride[0] = stride[0]; 
    maxpool->stride[1] = stride[1];

    int out_height = (maxpool->in_shape.height + 2 * maxpool->padding[0] - maxpool->kernel_sh.size[0]) / maxpool->stride[0] + 1;
    int out_width = (maxpool->in_shape.width + 2 * maxpool->padding[1] - maxpool->kernel_sh.size[1])   / maxpool->stride[1] + 1;
    
    maxpool->out_shape.batch_size = maxpool->in_shape.batch_size;
    maxpool->out_shape.height = out_height;     
    maxpool->out_shape.width = out_width;
    maxpool->out_shape.channels = in_shape.channels;
    for (int b = 0; b < maxpool->in_shape.batch_size; b++) {
        for (int ic = 0; ic < maxpool->in_shape.channels; ic++) {
            for (int oh = 0; oh < out_height; ++oh) {
                for (int ow = 0; ow < out_width; ++ow) {
                    float max_val = -FLT_MAX;
                    int max_idx = 0; //chỉ số của giá trị lớn nhất trong input
                    for (int kh = 0; kh < kernel_sh.size[0]; ++kh) {
                        for (int kw = 0; kw < kernel_sh.size[1]; ++kw) {
                            int ih = oh * stride[0] + kh - padding[0];
                            int iw = ow * stride[1] + kw - padding[1];
                            if (ih >= 0 && ih < maxpool->in_shape.height && iw >= 0 && iw < maxpool->in_shape.width) {
                                int input_idx = b * maxpool->in_shape.channels * maxpool->in_shape.height * maxpool->in_shape.width + 
                                                ic * maxpool->in_shape.height * maxpool->in_shape.width  + ih * maxpool->in_shape.width + iw;
                                float val = input[input_idx];
                                if (val > max_val) {
                                    max_val = val;
                                    max_idx = input_idx; // vị trí max
                                }
                            }
                        }
                    }
                    int output_idx = b * maxpool->in_shape.channels * out_height * out_width + 
                                    ic * out_height * out_width + 
                                    oh * out_width + ow;
                    output[output_idx] = max_val;
                    maxpool->max_indexList[output_idx] = max_idx; 
                }
            }
        }
    }
}

// Maxpool2 backward 

void Maxpool_backward(Layer* maxpool, float* output_gradient, float* input_gradient) {
    int input_size = maxpool->in_shape.batch_size * maxpool->in_shape.channels *
                     maxpool->in_shape.height * maxpool->in_shape.width;

    // Reset gradient input
    for (int i = 0; i < input_size; i++) {
        input_gradient[i] = 0;
    }

    int output_size = maxpool->out_shape.batch_size * maxpool->out_shape.channels *
                      maxpool->out_shape.height * maxpool->out_shape.width;

    for (int i = 0; i < output_size; i++) {
        int max_idx = maxpool->max_indexList[i];
        if (max_idx >= 0 && max_idx < input_size) {  // Kiểm tra giới hạn an toàn
            input_gradient[max_idx] += output_gradient[i];
        } else {
            printf("Warning: Invalid max_idx %d at output index %d\n", max_idx, i);
        }
    }
}

Feature_map_shape Layer_get_output_shape(const Layer* layer) { 
    return layer->out_shape; 
}

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

// Hàm fullyConnected_backward

void fullyConnected_backward(int num_NodePreLayer, int num_NodeThisLayer, int batch_size, Params params, bool use_bias, float* input, float* output_gradient, float* input_gradient, float learning_rate) {
    
    float* weights_gradient = (float*)calloc(num_NodeThisLayer * num_NodePreLayer, sizeof(float));
    if (weights_gradient == NULL) {
        printf("Error: Failed to allocate memory for weights_gradient!\n");
        return;
    }


    float* bias_gradient = NULL;
    if (use_bias) {
        bias_gradient = (float*)calloc(num_NodeThisLayer, sizeof(float));
        if (bias_gradient == NULL) {
            printf("Error: Failed to allocate memory for bias_gradient!\n");
            free(weights_gradient);
            return;
        }
    }


    for (int b = 0; b < batch_size; b++) {
        float* current_output_grad = output_gradient + (b * num_NodeThisLayer); // dL/dY
        float* current_input = input + (b * num_NodePreLayer); // X
        float* current_input_grad = input_gradient + (b * num_NodePreLayer); // dL/dX

        // (dL/dW) = dL/dY * X^T
        for (int i = 0; i < num_NodeThisLayer; i++) {
            for (int j = 0; j < num_NodePreLayer; j++) {
                int weight_idx = i * num_NodePreLayer + j;
                if (b == 0) {
                    weights_gradient[weight_idx] = current_output_grad[i] * current_input[j];
                } else {
                    weights_gradient[weight_idx] += current_output_grad[i] * current_input[j];
                }
            }
        }

        // (dL/dB) = dL/dY 
        if (use_bias) {
            for (int i = 0; i < num_NodeThisLayer; i++) {
                if (b == 0) {
                    bias_gradient[i] = current_output_grad[i];
                } else {
                    bias_gradient[i] += current_output_grad[i];
                }
            }
        }

        // (dL/dX) = W^T * dL/dY
        for (int i = 0; i < num_NodePreLayer; i++) {
            current_input_grad[i] = 0;
            for (int j = 0; j < num_NodeThisLayer; j++) {
                int weight_idx = j * num_NodePreLayer + i;
                current_input_grad[i] += params.weight[weight_idx] * current_output_grad[j];
            }    
        }
    }

    // W = W - learning_rate * dL/dW
    for (int i = 0; i < num_NodeThisLayer * num_NodePreLayer; i++) {
        params.weight[i] -= learning_rate * weights_gradient[i];
    }

    // B = B - learning_rate * dL/dB nếu use_bias = true
    if (use_bias) {
        for (int i = 0; i < num_NodeThisLayer; i++) {
            params.bias[i] -= learning_rate * bias_gradient[i];
        }
    }

    free(weights_gradient);
    if (use_bias) {
        free(bias_gradient);
    }
}

void Flatten_forward(float *input, float *output, Feature_map_shape shape) {
    int batch_size = shape.batch_size;
    int channels = shape.channels;
    int height = shape.height;
    int width = shape.width;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int index_input = b * channels * height * width + c * height * width + h * width + w;
                    int index_output = b * height * width * channels + h * width * channels + w * channels + c;
                    output[index_output] = input[index_input];
                }
            }
        }
    }
}

void FlattenBackward(float *grad_output, float *grad_input, Feature_map_shape shape) {
    int batch_size = shape.batch_size;
    int channels = shape.channels;
    int height = shape.height;
    int width = shape.width;

    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int index_output = b * height * width * channels + h * width * channels + w * channels + c;
                    int index_input = b * channels * height * width + c * height * width + h * width + w;
                    grad_input[index_input] = grad_output[index_output];
                }
            }
        }
    }
}


// Hàm đọc dữ liệu từ file cho Conv2D
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
        //printf("Sample %d - Predicted class: %s (index: %d)\n", i, class_names[predict], predict);
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
