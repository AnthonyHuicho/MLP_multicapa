#include "data_loader.h"
#include <fstream>
#include <iostream>
                
std::vector<CIFARImage> load_cifar10_batch(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) { std::cerr << "No se pudo abrir " << filename << std::endl; exit(1); }

    std::vector<CIFARImage> batch;
    CIFARImage img;
    while (file.read((char*)&img.label, 1)) {
        file.read((char*)img.pixels, 3072);
        batch.push_back(img);
    }
    return batch;
}