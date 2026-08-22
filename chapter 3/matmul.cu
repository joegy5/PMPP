#include <stdlib.h>
#include <stdio.h>
#include <iostream> 
using namespace std;

__global__
void matmulKernel(float* A, float* B, float* C, int n, int k, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if(row < n && col < m) {
        int innerProd = 0; 
        for(int i = 0; i < k; i++) {
            innerProd += A[row * m + i] * B[i * m + col];
        }

        C[row * m + col] = innerProd; 
    }
}

__host__
void matmul(float* A_h, float* B_h, float* C_h, int n, int k, int m) { 
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, n * k * sizeof(float));
    cudaMalloc((void**)&B_d, k * m * sizeof(float));
    cudaMalloc((void**)&C_d, n * m * sizeof(float)); 

    cudaMemcpy(A_d, A_h, n * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, k * m * sizeof(float), cudaMemcpyDeviceToHost);
    
    dim3 dimGrid(ceil(m / 16.0), ceil(n / 16.0), 1);
    dim3 dimBlock(16, 16, 1); 
    matmulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n, k, m); 

    cudaMemcpy(C_h, C_d, n * m * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d); 
    cudaFree(C_d); 
}

int main() {
    float A[2][2] = {{2, 2}, {2, 2}};
    float B[2][2] = {{2, 2}, {2, 2}}; 
    float C[2][2]; 
    matmul((float*)A, (float*)B, (float*)C, 2, 2, 2); 
    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            cout << C[i][j] << " "; 
        }
        cout << endl;
    }
}