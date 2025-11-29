#pragma once
#include <vector>
#include <string>
struct CIFARImage {
    unsigned char label;
    unsigned char pixels[3072];
};

std::vector<CIFARImage> load_cifar10_batch(const std::string& filename);