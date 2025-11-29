#include "data_loader.h"
#include "perceptron.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <cmath>


#define N_PIXELS 3072      
#define N_CLASSES 10       
#define N_IMAGES 10000    

#define ETA 0.3f          
#define MOMENTO 0.9f      
#define N_EPOCHS 150       // épocas 
#define TOLERANCIA 0.001f 

#define N_HIDDEN_LAYERS 3  // 3 capas ocultas
#define HIDDEN_SIZE_1 4096 // neuronas
#define HIDDEN_SIZE_2 2048  //  neuronas
#define HIDDEN_SIZE_3 1024  //  neuronas

const char* CLASS_NAMES[10] = {
    "avion", "auto", "pajaro", "gato", "ciervo",
    "perro", "rana", "caballo", "barco", "camion"
};

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
        << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
        exit(1); \
    }

struct Layer {
    float* d_weights;
    float* d_weights_prev;
    float* d_bias;
    float* d_bias_prev;
    float* d_output;
    float* d_u;
    float* d_error;
    int input_size;
    int output_size;
};

void initialize_layer(Layer& layer, int input_size, int output_size, int batch_size) {
    layer.input_size = input_size;
    layer.output_size = output_size;

    CUDA_CHECK(cudaMalloc(&layer.d_weights, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_weights_prev, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_bias, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_bias_prev, output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_output, batch_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_u, batch_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer.d_error, batch_size * output_size * sizeof(float)));

    std::vector<float> h_weights(input_size * output_size);
    std::vector<float> h_bias(output_size);

    float scale = sqrtf(2.0f / input_size);
    for (auto& w : h_weights) {
        w = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    for (auto& b : h_bias) {
        b = 0.01f;
    }

    CUDA_CHECK(cudaMemcpy(layer.d_weights, h_weights.data(),
        h_weights.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer.d_bias, h_bias.data(),
        h_bias.size() * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(layer.d_weights_prev, 0, input_size * output_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(layer.d_bias_prev, 0, output_size * sizeof(float)));
}

void free_layer(Layer& layer) {
    cudaFree(layer.d_weights);
    cudaFree(layer.d_weights_prev);
    cudaFree(layer.d_bias);
    cudaFree(layer.d_bias_prev);
    cudaFree(layer.d_output);
    cudaFree(layer.d_u);
    cudaFree(layer.d_error);
}


int predict_cpu(const std::vector<std::vector<float>>& weights,
    const std::vector<std::vector<float>>& biases,
    const std::vector<float>& input)
{
    int layer_sizes[] = { N_PIXELS, HIDDEN_SIZE_1, HIDDEN_SIZE_2, HIDDEN_SIZE_3, N_CLASSES };
    int n_layers = weights.size();
    std::vector<float> current_input = input;

    for (int l = 0; l < n_layers; ++l) {
        int input_size = layer_sizes[l];
        int output_size = layer_sizes[l + 1];

        std::vector<float> output(output_size);

        for (int j = 0; j < output_size; ++j) {
            float u = biases[l][j];
            for (int i = 0; i < input_size; ++i) {
                u += weights[l][j * input_size + i] * current_input[i];
            }

            if (l < n_layers - 1) {
                output[j] = fmaxf(0.0f, u);
            }
            else {
                output[j] = u;
            }
        }

        current_input = output;
    }

    int pred = 0;
    float max_val = current_input[0];
    for (int i = 1; i < N_CLASSES; ++i) {
        if (current_input[i] > max_val) {
            max_val = current_input[i];
            pred = i;
        }
    }

    return pred;
}

void shuffle_data(std::vector<float>& images, std::vector<unsigned char>& labels) {
    std::vector<int> indices(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) indices[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<float> temp_images = images;
    std::vector<unsigned char> temp_labels = labels;

    for (int i = 0; i < N_IMAGES; ++i) {
        for (int j = 0; j < N_PIXELS; ++j) {
            images[i * N_PIXELS + j] = temp_images[indices[i] * N_PIXELS + j];
        }
        labels[i] = temp_labels[indices[i]];
    }
}



int main() {
    srand(time(NULL));

    std::cout << "  Mlp\n";


    auto batch = load_cifar10_batch("C:/Users/Usuario/Downloads/cifar-10-binary/cifar-10-batches-bin/data_batch_1.bin");

    std::vector<float> h_images(N_IMAGES * N_PIXELS);
    std::vector<unsigned char> h_labels(N_IMAGES);

    for (int i = 0; i < N_IMAGES; ++i) {
        h_labels[i] = batch[i].label;
        for (int j = 0; j < N_PIXELS; ++j) {
            h_images[i * N_PIXELS + j] = (batch[i].pixels[j] / 127.5f) - 1.0f; 
        }
    }

    float* d_images;
    unsigned char* d_labels;
    CUDA_CHECK(cudaMalloc(&d_images, h_images.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels, h_labels.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int total_layers = N_HIDDEN_LAYERS + 1;
    std::vector<Layer> layers(total_layers);

    std::cout << "Creando red neuronal:\n";
    std::cout << "  Entrada: " << N_PIXELS << " neuronas\n";

    initialize_layer(layers[0], N_PIXELS, HIDDEN_SIZE_1, N_IMAGES);
    std::cout << "  Capa oculta 1: " << HIDDEN_SIZE_1 << " neuronas\n";

    initialize_layer(layers[1], HIDDEN_SIZE_1, HIDDEN_SIZE_2, N_IMAGES);
    std::cout << "  Capa oculta 2: " << HIDDEN_SIZE_2 << " neuronas\n";

    initialize_layer(layers[2], HIDDEN_SIZE_2, HIDDEN_SIZE_3, N_IMAGES);
    std::cout << "  Capa oculta 3: " << HIDDEN_SIZE_3 << " neuronas\n";

    initialize_layer(layers[N_HIDDEN_LAYERS], HIDDEN_SIZE_3, N_CLASSES, N_IMAGES);
    std::cout << "  Capa de salida: " << N_CLASSES << " neuronas\n\n";

    int threads = 256;

    std::cout << "Parametros de entrenamiento:\n";
    std::cout << "tasa de aprendizaje: " << ETA << "\n";
    std::cout << "Epocas: " << N_EPOCHS << "\n";

    float best_accuracy = 0.0f;
    int epochs_without_improvement = 0;

    for (int epoch = 0; epoch < N_EPOCHS; ++epoch) {
        shuffle_data(h_images, h_labels);
        CUDA_CHECK(cudaMemcpy(d_images, h_images.data(), h_images.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_labels, h_labels.data(), h_labels.size() * sizeof(unsigned char), cudaMemcpyHostToDevice));

        const float* current_input = d_images;

        for (int l = 0; l < total_layers - 1; ++l) {
            int blocks = (N_IMAGES * layers[l].output_size + threads - 1) / threads;

            layer_forward_with_relu << <blocks, threads >> > (
                layers[l].d_weights,
                layers[l].d_bias,
                current_input,
                layers[l].d_output,
                layers[l].d_u,
                layers[l].input_size,
                layers[l].output_size,
                N_IMAGES
                );
            CUDA_CHECK(cudaDeviceSynchronize());

            current_input = layers[l].d_output;
        }

        int l = total_layers - 1;
        int blocks = (N_IMAGES * layers[l].output_size + threads - 1) / threads;
        output_layer_forward << <blocks, threads >> > (
            layers[l].d_weights,
            layers[l].d_bias,
            current_input,
            layers[l].d_output,
            layers[l].input_size,
            layers[l].output_size,
            N_IMAGES
            );
        CUDA_CHECK(cudaDeviceSynchronize());

        int blocks_out = (N_IMAGES * N_CLASSES + threads - 1) / threads;
        output_layer_backward << <blocks_out, threads >> > (
            layers[total_layers - 1].d_output,
            d_labels,
            layers[total_layers - 1].d_error,
            N_IMAGES,
            N_CLASSES
            );
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int l = total_layers - 1; l >= 1; --l) {
            int blocks_back = (N_IMAGES * layers[l - 1].output_size + threads - 1) / threads;
            layer_backward << <blocks_back, threads >> > (
                layers[l].d_weights,
                layers[l].d_error,
                layers[l - 1].d_u,
                layers[l - 1].d_error,
                layers[l].input_size,
                layers[l].output_size,
                N_IMAGES
                );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        int blocks_update0 = (layers[0].input_size * layers[0].output_size + threads - 1) / threads;
        update_weights << <blocks_update0, threads >> > (
            layers[0].d_weights,
            layers[0].d_bias,
            layers[0].d_error,
            d_images,
            layers[0].input_size,
            layers[0].output_size,
            N_IMAGES,
            ETA
            );
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int l = 1; l < total_layers; ++l) {
            int blocks_update = (layers[l].input_size * layers[l].output_size + threads - 1) / threads;
            update_weights << <blocks_update, threads >> > (
                layers[l].d_weights,
                layers[l].d_bias,
                layers[l].d_error,
                layers[l - 1].d_output,
                layers[l].input_size,
                layers[l].output_size,
                N_IMAGES,
                ETA
                );
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        std::vector<float> h_output(N_IMAGES * N_CLASSES);
        CUDA_CHECK(cudaMemcpy(h_output.data(), layers[total_layers - 1].d_output,
            h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        int correct = 0;
        double total_error = 0.0;

        for (int i = 0; i < N_IMAGES; ++i) {
            float max_val = h_output[i * N_CLASSES];
            int pred = 0;

            for (int c = 1; c < N_CLASSES; ++c) {
                if (h_output[i * N_CLASSES + c] > max_val) {
                    max_val = h_output[i * N_CLASSES + c];
                    pred = c;
                }
            }

            for (int c = 0; c < N_CLASSES; ++c) {
                float d = (h_labels[i] == c) ? 1.0f : 0.0f;
                float y = h_output[i * N_CLASSES + c];
                total_error += (d - y) * (d - y) / 2.0;
            }

            if (pred == h_labels[i]) correct++;
        }

        float accuracy = (float)correct / N_IMAGES;
        total_error /= N_IMAGES;

        std::cout << "Epoca " << (epoch + 1) << "/" << N_EPOCHS
            << " - Accuracy: " << (accuracy) << ""
            << " - Error: " << total_error << std::endl;

        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            epochs_without_improvement = 0;
        }
        else {
            epochs_without_improvement++;
        }
        if (epochs_without_improvement >= 40 && epoch > 50) {
            std::cout << "\nEarly stopping - Sin mejora en 40 epocas\n";
            std::cout << "Mejor accuracy alcanzado: " << (best_accuracy) << "\n";
            break;
        }

        if (total_error < TOLERANCIA) {
            std::cout << "\nConvergio\n";
            break;
        }
    }


    std::cout << "\n";
    std::cout << "final\n";
    std::cout << "Mejor accuracy: " << (best_accuracy) << "\n\n";

    std::vector<float> h_output(N_IMAGES * N_CLASSES);
    CUDA_CHECK(cudaMemcpy(h_output.data(), layers[total_layers - 1].d_output,
        h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));

    int correct = 0;
    for (int i = 0; i < N_IMAGES; ++i) {
        float max_val = h_output[i * N_CLASSES];
        int pred = 0;

        for (int c = 1; c < N_CLASSES; ++c) {
            if (h_output[i * N_CLASSES + c] > max_val) {
                max_val = h_output[i * N_CLASSES + c];
                pred = c;
            }
        }

        if (pred == h_labels[i]) correct++;
    }

    std::cout << "Precision final: " << (100.0f * correct / N_IMAGES) << "%\n";


    std::cout << "\n";
    std::cout << "PRUEBA CON IMAGEN INDIVIDUAL\n";
    std::cout << "\n";

    int test_idx = 5;

    std::vector<std::vector<float>> cpu_weights(total_layers);
    std::vector<std::vector<float>> cpu_biases(total_layers);

    for (int l = 0; l < total_layers; ++l) {
        int w_size = layers[l].input_size * layers[l].output_size;
        cpu_weights[l].resize(w_size);
        cpu_biases[l].resize(layers[l].output_size);

        CUDA_CHECK(cudaMemcpy(cpu_weights[l].data(), layers[l].d_weights,
            w_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cpu_biases[l].data(), layers[l].d_bias,
            layers[l].output_size * sizeof(float), cudaMemcpyDeviceToHost));
    }

    std::vector<float> test_img(N_PIXELS);
    for (int j = 0; j < N_PIXELS; ++j)
        test_img[j] = h_images[test_idx * N_PIXELS + j];

    int pred = predict_cpu(cpu_weights, cpu_biases, test_img);

    std::cout << "Imagen #" << test_idx << "\n";
    std::cout << "  Etiqueta real: " << (int)h_labels[test_idx]
        << " (" << CLASS_NAMES[h_labels[test_idx]] << ")\n";
    std::cout << "  Predicción:    " << pred
        << " (" << CLASS_NAMES[pred] << ")\n";
    std::cout << "  Resultado: " << (pred == h_labels[test_idx] ? "CORRECTO" : " INCORRECTO") << "\n";

    for (auto& layer : layers) {
        free_layer(layer);
    }
    cudaFree(d_images);
    cudaFree(d_labels);

    return 0;
}