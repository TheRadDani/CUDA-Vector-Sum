// vecAdd.h
#ifndef VECADD_H
#define VECADD_H

#include <iostream>
#include <cuda_runtime.h>

// Function to perform vector addition on the GPU
__global__ void vecAdd(double *a, double *b, double *c, int n);

#endif // VECADD_H
