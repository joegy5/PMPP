#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#define TILE_WIDTH 4
__global__ void tiledMatMulKernel(float* A, float* B, float* C, int m, int k, int n) {
    // each block calculates a T x T tile of the output matrix, where T is the tile width 
    __shared__ float M[TILE_WIDTH][TILE_WIDTH]; // NOTE: TILE_WIDTH has to be constant, can't declare arrays with variable lengths
    __shared__ float N[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float res = 0; // re-using same res value in register for each phase -> REGISTER TILING
    
    // strip-mining: take long-running loop and break into phases
    for(int i = 0; i < ceil(k / (float)TILE_WIDTH); i++) {
        // load the shared values inot the M and N arrays
        //  - thread covering position (i, j) of the output tile loads 
        //    T_A[i][j] and T_B[i][j], where T_A and T_B are the input matrix tiles of
        //    the current phase
            
        // boundary checks - only need to check if bottom or right sides are out of bounds, not left or top sides
        if((i * blockDim.x + threadIdx.x) < k && row < m) {
            M[threadIdx.y][threadIdx.x] = A[row * k + i * blockDim.x + threadIdx.x]; 
            
        } else {
            M[threadIdx.y][threadIdx.x] = 0.0f; // force float, not double
        }
        if((i * blockDim.y + threadIdx.y) < k && col < n) {
            N[threadIdx.y][threadIdx.x] = B[(i * blockDim.y + threadIdx.y) * n + col];
        } else {
            N[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // read-after-write dependency: threads must wait for data to be written before processing it
        __syncthreads(); // ensure all values loaded into M and N before we can matmul them
            
        for(int j = 0; j < TILE_WIDTH; j++) {
            res += M[threadIdx.y][j] * N[j][threadIdx.x];
        }
        // write-after-read dependency: thread must wait for data to be read by all threads needing it, before overwriting it
        __syncthreads(); // make sure matmul completely finished before overwriting tiles with next phase's values
        

    }

    // NOTE: do NOT check this at the very beginning; even if out of bounds, we still need corresponding
    //  positions in M and N matrices to be written as zeros, otherwise can lead to undefined behavior
    if(row < m && col < n) {
        C[row * n + col] = res;
    } 
}

__host__ void tiledMatMul(float* A_h, float* B_h, float* C_h, int m, int k, int n) {
    float *A_d, *B_d, *C_d;
    int A_size = m * k * sizeof(float);
    int B_size = k * n * sizeof(float);
    int C_size = m * n * sizeof(float);

    cudaMalloc((void**)&A_d, A_size);
    cudaMalloc((void**)&B_d, B_size);
    cudaMalloc((void**)&C_d, C_size); 

    cudaMemcpy(A_d, A_h, A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, B_size, cudaMemcpyHostToDevice);
    
    // A: m x k, B: k x n
    
    dim3 dimGrid(ceil(n / (float)TILE_WIDTH), ceil(m / (float)TILE_WIDTH), 1); // ordered from least to greatest dim (x -> z)
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1); // ordered from least to greatest dim (x -> z)
    tiledMatMulKernel<<<dimGrid, dimBlock>>>(
        A_d, B_d, C_d,
        m, k, n
    ); 

    cudaMemcpy(C_h, C_d, C_size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d); 
}

int main() {
    int m = 6, k = 5, n = 7; 

    float A[m][k];
    float B[k][n];
    float C[m][n];
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < k; j++) {
            A[i][j] = i * k + j * 1.0;
            cout << A[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
    for(int i = 0; i < k; i++) {
        for(int j = 0; j < n; j++) {
            B[i][j] = i * n + j * 1.0;
            cout << B[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;

    tiledMatMul((float*)A, (float*)B, (float*)C, m, k, n); 
    
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            cout << C[i][j] << " ";
        }
        cout << endl; 
    }
}  