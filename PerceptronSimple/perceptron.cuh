#ifndef PERCEPTRON_CUH
#define PERCEPTRON_CUH

__global__ void layer_forward_with_relu(
    const float* d_weights,
    const float* d_bias,
    const float* d_input,
    float* d_output,
    float* d_pre_activation,
    int input_size,
    int output_size,
    int batch_size);


__global__ void output_layer_forward(
    const float* d_weights,
    const float* d_bias,
    const float* d_input,
    float* d_output,
    int input_size,
    int output_size,
    int batch_size);

__global__ void output_layer_backward(
    const float* d_output,
    const unsigned char* d_labels,
    float* d_grad_output,
    int batch_size,
    int n_classes);

__global__ void layer_backward(
    const float* d_weights,
    const float* d_grad_output,
    const float* d_pre_activation,
    float* d_grad_input,
    int input_size,
    int output_size,
    int batch_size);

__global__ void update_weights(
    float* d_weights,
    float* d_bias,
    const float* d_grad_output,
    const float* d_input,
    int input_size,
    int output_size,
    int batch_size,
    float eta);

#endif