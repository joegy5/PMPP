#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__
void imageBlurKernel(unsigned char* Pin, unsigned char* Pout, int n, int m, int BLUR_SIZE) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < n && col < m) {
        int blur_radius = BLUR_SIZE / 2; 
        int pixelSum = 0; // use int, since summing could go beyond 0-255 range
        int numPixels = 0; 
        for(int blur_row = -blur_radius; blur_row < blur_radius + 1; blur_row++) {
            for(int blur_col = -blur_radius; blur_col < blur_radius + 1; blur_col++) {
                int currRow = row + blur_row;
                int currCol = col + blur_col; 
                if(currRow >= 0 && currRow < n && currCol >= 0 && currCol < m) { 
                    pixelSum += Pin[currRow * m + currCol]; 
                    numPixels++; 
                }
            }

        }
        Pout[row * m + col] = (unsigned char) ((float) pixelSum / numPixels);
    } 
}

__host__
void imageBlur(unsigned char* Pin_h, unsigned char* Pout_h, int n, int m) {
    // allocate device memory for Pin_d and Pout_d
    int size = n * m * sizeof(unsigned char); 
    unsigned char *Pin_d, *Pout_d;

    cudaMalloc((void**)&Pin_d, size);
    cudaMalloc((void**)&Pout_d, size);

    // copy Pin to device memory
    cudaMemcpy(Pin_d, Pin_h, size, cudaMemcpyHostToDevice);

    // launch image blur kernel
    dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
    dim3 dimBlock(16, 16, 1);
    int BLUR_SIZE = 3; // must be an odd number to make sense 
    imageBlurKernel<<<dimGrid, dimBlock>>>(Pin_d, Pout_d, n, m, BLUR_SIZE);  

    // copy Pout to host memory
    cudaMemcpy(Pout_h, Pout_d, size, cudaMemcpyDeviceToHost);
    
    // free device memory
    cudaFree(Pin_d);
    cudaFree(Pout_d); 
}

int main() {
    unsigned char Pin[3][3] = {{1, 2, 3}, {2, 3, 4}, {5, 6, 7 }};
    unsigned char Pout[3][3]; 

    imageBlur((unsigned char*)Pin, (unsigned char*)Pout, 3, 3); 
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            cout << (int)Pout[i][j] << " "; 
        }
        cout << endl; 
    }
}