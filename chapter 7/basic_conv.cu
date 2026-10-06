#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

__global__ void basic_conv_kernel(float* I, float* F, float* O, int m, int n, int f_radius) {
    // get output position of the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    int filter_dim = 2 * f_radius + 1;

    float res = 0.0;
    for(int i = 0; i < filter_dim; i++) {
        for(int j = 0; j < filter_dim; j++) {
            int i_coord = (i + row - f_radius), j_coord = j + col - f_radius;
            if(i_coord >= 0 && i_coord < m && j_coord >= 0 && j_coord < n) {
                res += I[i_coord * n + j_coord] * F[i * filter_dim + j];
            }
        }
    }

    O[row * n + col] = res;
}

void basic_conv(float* I_h, float* F_h, float* O_h, int m, int n, int block_sz_row, int block_sz_col, int f_radius) {
    // allocate device memory for A and F
    int I_sz = m * n * sizeof(float); 
    int F_sz = (2 * f_radius + 1) * (2 * f_radius + 1) * sizeof(float); 
    float *I_d, *F_d, *O_d; 

    cudaMalloc((void**)&I_d, I_sz);
    cudaMalloc((void**)&O_d, I_sz); 
    cudaMalloc((void**)&F_d, F_sz); 

    // copy host memory variables into allocated device memory
    cudaMemcpy(I_d, I_h, I_sz, cudaMemcpyHostToDevice); 
    cudaMemcpy(F_d, F_h, F_sz, cudaMemcpyHostToDevice); 

    // perform the computation
    dim3 dimGrid(ceil(n / (float)block_sz_col), ceil(m / (float)block_sz_row), 1);
    dim3 dimBlock(block_sz_col, block_sz_row, 1); // each thread computes one element in the output matrix
    basic_conv_kernel<<<dimGrid, dimBlock>>>(I_d, F_d, O_d, m, n, f_radius); 

    // copy output in device memory back to host memory
    cudaMemcpy(O_h, O_d, I_sz, cudaMemcpyDeviceToHost); 

    // free the allocated device memory
    cudaFree(I_d);
    cudaFree(O_d);
    cudaFree(F_d); 
    
}

int main() {
    // create the input arrays, as well as the output array
    int num_rows = 1 << 12, num_cols = 1 << 12; 
    int block_sz_row = 10, block_sz_col = 10; 
    int f_radius = 1; 
    float* I = (float*)malloc(num_rows * num_cols * sizeof(float)); 
    float* O = (float*)malloc(num_rows * num_cols * sizeof(float)); 
    float* F = (float*)malloc((2 * f_radius + 1) * (2 * f_radius + 1) * sizeof(float));

    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            I[i * num_cols + j] = 1; 
        }
    }
    for(int i = 0; i < 2 * f_radius + 1; i++) {
        for(int j = 0; j < 2 * f_radius + 1; j++) {
            F[i * (2 * f_radius + 1) + j] = 1; 
        }
    }

    // call the kernel function
    basic_conv((float*)I, (float*)F, (float*)O, num_rows, num_cols, block_sz_row, block_sz_col, f_radius);

    // print out the results
    // for(int i = 0; i < num_rows; i++) {
    //     for(int j = 0; j < num_cols; j++) {
    //         cout << O[i * num_cols + j] << " ";
    //     }
    //     cout << endl; 
    // }
}