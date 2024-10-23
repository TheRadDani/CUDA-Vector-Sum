// main.cu
#include "vecAdd.h"
#include <math.h>
#include <cassert>

int main(int argc, const char **argv) {
    // Size of vectors
    int n = 1000;
    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;

    // Size in bytes for each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on CPU
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    for(int i=0; i<n; i++) {
        h_a[i] = rand() % 10000;
        h_b[i] = rand() % 10000;
    }

    for(int i=0; i<n; i++) {
        h_c[i] = h_a[i] + h_b[i];
    }

    for(int i=0; i<n; i++) {
        assert(h_a[i] + h_b[i] == h_c[i]);
    }

    // Allocate memory for each vector on GPU
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Number of threads per block
    int blockSize = 1024;

    // Number of thread blocks in the grid
    int gridSize = (int)ceil((float)n / blockSize);

    // Execute the kernel
    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Synchronize device
    cudaDeviceSynchronize();

    // Copy result vector back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    // Sum up the vector c and print the result divided by n to normalize
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += h_c[i];
    }
    double result = sum / n;

    std::cout << "Result: " << result << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
