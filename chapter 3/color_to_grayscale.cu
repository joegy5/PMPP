#include <stdlib.h>
#include <stdio.h>
#include <iostream> 
using namespace std;

__global__
void colorToGrayscaleKernel(unsigned char* Pin, unsigned char* Pout, int n, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < n && col < m) {
        int gray_idx = row * m + col; 
        // can think of each row in Pin having m * numChannels columns, due to row-major layout
        int rgb_idx = gray_idx * 3; 
        unsigned char red = Pin[rgb_idx];
        unsigned char green = Pin[rgb_idx + 1]; 
        unsigned char blue = Pin[rgb_idx + 2];
        // type cast back to unsigned char since we are multiplying by float values
        // suffix 'f' indicates that the constants are floats, not doubles
        // NOTE: this truncates the decimal value of the result, not round up/down
        Pout[gray_idx] = (unsigned char) (0.21f * red + 0.71f * green + 0.07f * blue);
    }
}

// NOTE: though Pin_h and Pout_h are 2D arrays, in actual memory they are just 1D array of bytes
//  therefore, we can use a pointer pointing to the 0th element in the array, like with 1D arrays
__host__
void colorToGrayscale(unsigned char* Pin_h, unsigned char* Pout_h, int n, int m) {
    // Allocate device memory for P_in and P_out
    unsigned char *Pin_d, *Pout_d; 
    int size = n * m * sizeof(unsigned char);

    cudaMalloc((void**)&Pin_d, size * 3);
    cudaMalloc((void**)&Pout_d, size);

    // Copy memory from Pin_h to device (no need to do for Pout_d, since it's already on device as empty)
    cudaMemcpy(Pin_d, Pin_h, size * 3, cudaMemcpyHostToDevice);

    // Launch kernel
    // order: x, y, z (least to greatest dimension); greatest to least dimension order would be z, y, x
    dim3 dimGrid(ceil(m / (16.0)), ceil(n / 16.0), 1); // ceil(m / 16) blocks in x dimension, ceil(n / 16) blocks in y dimension, 1 block ion z dimension (only using x and y dimensions)
    dim3 dimBlock(16, 16, 1); // 16 threads in x dimension as well as y dimension, 1 thread in z dimension (only using x and y dimensions)
    colorToGrayscaleKernel<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, n, m);

    // Copy Pin_d from device memory to host memory
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
    // Free device memory 
    cudaFree(Pin_d);
    cudaFree(Pout_d); 
}

int main() {
    unsigned char Pin[2][2][3] = {{{1, 1, 1}, {1, 1, 1}},{{1, 1, 1}, {1, 1, 1}}};
    unsigned char Pout[2][2]; 
    // Pin and Pout arrays are contiguous in memory -> can cast them as unsigned char* 
    colorToGrayscale((unsigned char*) Pin, (unsigned char*) Pout, 2, 2); 
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            cout << (int)Pout[i][j] << " ";
        }
        cout << endl; 
    }
}