#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <math.h>


__global__ void layer_forward_with_relu(
    const float* d_weights,
    const float* d_bias,
    const float* d_input,
    float* d_output,
    float* d_pre_activation,
    int input_size,
    int output_size,
    int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_size) return;

    int sample = idx / output_size;
    int neuron = idx % output_size;

    float sum = d_bias[neuron];
    for (int i = 0; i < input_size; ++i) {
        sum += d_weights[neuron * input_size + i] * d_input[sample * input_size + i];
    }

   
    d_pre_activation[idx] = sum;


    d_output[idx] = fmaxf(0.0f, sum);
}


__global__ void output_layer_forward(
    const float* d_weights,
    const float* d_bias,
    const float* d_input,
    float* d_output,
    int input_size,
    int output_size,
    int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_size) return;

    int sample = idx / output_size;
    int neuron = idx % output_size;

  
    float sum = d_bias[neuron];
    for (int i = 0; i < input_size; ++i) {
        sum += d_weights[neuron * input_size + i] * d_input[sample * input_size + i];
    }

    d_output[idx] = sum;
}


__global__ void output_layer_backward(
    const float* d_output,
    const unsigned char* d_labels,
    float* d_grad_output,
    int batch_size,
    int n_classes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * n_classes) return;

    int sample = idx / n_classes;
    int class_idx = idx % n_classes;

  
    float max_val = d_output[sample * n_classes];
    for (int c = 1; c < n_classes; ++c) {
        float val = d_output[sample * n_classes + c];
        if (val > max_val) max_val = val;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < n_classes; ++c) {
        sum_exp += expf(d_output[sample * n_classes + c] - max_val);
    }

    float softmax = expf(d_output[idx] - max_val) / sum_exp;
    float target = (d_labels[sample] == class_idx) ? 1.0f : 0.0f;


    d_grad_output[idx] = softmax - target;
}


__global__ void layer_backward(
    const float* d_weights,
    const float* d_grad_output,
    const float* d_pre_activation,
    float* d_grad_input,
    int input_size,
    int output_size,
    int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * input_size) return;

    int sample = idx / input_size;
    int input_idx = idx % input_size;


    float grad = 0.0f;
    for (int o = 0; o < output_size; ++o) {
        grad += d_grad_output[sample * output_size + o] * d_weights[o * input_size + input_idx];
    }


    float relu_derivative = (d_pre_activation[idx] > 0.0f) ? 1.0f : 0.0f;
    d_grad_input[idx] = grad * relu_derivative;
}


__global__ void update_weights(
    float* d_weights,
    float* d_bias,
    const float* d_grad_output,
    const float* d_input,
    int input_size,
    int output_size,
    int batch_size,
    float eta)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= output_size * input_size) return;

    int neuron = idx / input_size;
    int input_idx = idx % input_size;


    float grad_sum = 0.0f;
    for (int s = 0; s < batch_size; ++s) {
        grad_sum += d_grad_output[s * output_size + neuron] * d_input[s * input_size + input_idx];
    }

 
    d_weights[idx] -= eta * grad_sum / batch_size;


    if (input_idx == 0) {
        float bias_grad = 0.0f;
        for (int s = 0; s < batch_size; ++s) {
            bias_grad += d_grad_output[s * output_size + neuron];
        }
        d_bias[neuron] -= eta * bias_grad / batch_size;
    }
}