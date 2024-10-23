// vecAdd.cu
#include "vecAdd.h"

// CUDA kernel for vector addition
__global__ void vecAdd(double *a, double *b, double *c, int n) {
    // Get global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we do not go out of bounds
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}
