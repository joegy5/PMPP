#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

// __host__: used for traditional C++ functions
// __device__: used for functions executed on GPU
// NOTE: both __host__ and __device__ can be used for single function 
//  - compiler generates two versions of object code for the same function 
//      - one version can only be called from another host function
//      - the other can only be called from another device or kernel function
__global__ // specifies that this is a kernel function
void vecAddKernel(float* A, float* B, float* C, int n) {
    // each block consists of certain number of threads (e.g. 256, 1024)
    // each kernel has block idx and thread idx (built-in variables)
    // kernel also has blockDim struct variable (# of threads per block)
    //  - blockDim.x: # threads in x dimension
    //  - blockDim.y: # threads in y dimension
    //  - blockDim.z: # threads in z dimension
    //  - note that these dimensions also apply to threadIdx and blockIdx
    // threadIdx and blockIdx together uniquely identify each launched kernel
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) { // since n might not be a multiple blockDim.x
        // A[i]: move pointer from memory addr A points to, to i * 4 bytes away, and dereference
        C[i] = A[i] + B[i]; 
    }
}

void vecAdd(float* A_h, float* B_h, float* C_h, int n) {
    // PART 1: Allocate device memory for A, B, and C
    int size = n * sizeof(float); // size of the input
    float *A_d, *B_d, *C_d; 
    // allocate memory from the GPU device for each pointer (in bytes)
    // each pointer has to be cast to void** (cudaMalloc expects generic pointer so it can assign memory, regardless of type)
    // we cast the ADDRESS of each pointer, not the pointer itself to void**
    // &A_d is of type float** (pointer to a pointer, since it's pointing to the addr of A_d, which itself is a pointer
    //  - "&" operator designed to automatically return pointer to the variable the symbol is added to
    //      - &anything returns memory address of "anything", but as pointer type, not just raw addr number
    //  - simply changing &A_d from float** to void** 
    cudaMalloc((void**)&A_d, size); 
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    // Copy A and B to device memory (analogous to tl.load() in Triton, but done in host device instead of GPU)
    // Don't need to pass &pointer, since we are concerned with just it's value, a.k.a the memory addresses the pointers point to 
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // PART 2: Call kernel to launch grid of threads that perform vector addition in parallel
    // ceil(n / 256.0) blocks, each having 256 threads
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n); 

    // PART 3: Copy C from device memory (similar to copying outputs in Triton)
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    // free device memory previously used by device vectors A_d, B_d, C_d
    // no need to pass &A_d, since we just need the value of A_d, a.k.a the memory address it is pointing to
    //  - once cudaFree has that address, it can free the memory at that address
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d); 
}

int main() {
    float A[5] = {1, 2, 3, 4, 5};
    float B[5] = {2, 3, 4, 5, 6};
    float C[5]; 
    int n = 5;
    vecAdd(A, B, C, n);
    for(int i = 0; i < n; i++) {
        cout << C[i] << " "; 
    }
    cout << endl; 
}