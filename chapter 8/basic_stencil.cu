#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#define BLOCK_PLANE_SIZE 1
#define BLOCK_ROW_SIZE 1
#define BLOCK_COL_SIZE 1

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

__global__ void basic_stencil_kernel(float* I, float* O, int l, int m, int n) {
    // each thread within the block computes one output element
    //printf("reached here\n");
    int plane = blockIdx.z * blockDim.z + threadIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    
   

    // check that the current position is in bounds AND not on edge/corner
    //printf("coords: (%d, %d, %d)\n", plane, row, col);

    if(plane >= 1 && plane < l - 1 && row >= 1 && row < m - 1 && col >= 1 && col < n - 1) {
        //printf("coords: (%d, %d, %d)\n", plane, row, col);
        O[plane * m * n + row * n + col] = c0 * I[plane * m * n + row * n + col] 
                                         + c1 * I[(plane - 1) * m * n + row * n + col]
                                         + c2 * I[(plane + 1) * m * n + row * n + col]
                                         + c3 * I[plane * m * n + (row - 1) * n + col]
                                         + c4 * I[plane * m * n + (row + 1) * n + col]
                                         + c5 * I[plane * m * n + row * n + (col - 1)]
                                         + c6 * I[plane * m * n + row * n + (col + 1)];
    }
}

void basic_stencil(float* I_h, float* O_h, int l, int m, int n) {
    int I_sz = l * m * n * sizeof(float);
    float *I_d, *O_d;

    cudaMalloc((void**)&I_d, I_sz);
    cudaMalloc((void**)&O_d, I_sz); 

    cudaMemcpy(I_d, I_h, I_sz, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil(n / (float)BLOCK_COL_SIZE), ceil(m / (float)BLOCK_ROW_SIZE), ceil(l / (float)BLOCK_PLANE_SIZE));
    dim3 dimBlock(BLOCK_COL_SIZE, BLOCK_ROW_SIZE, BLOCK_PLANE_SIZE);
    // printf("reached here\n"); 
    // cout << n << " " << m << " " << l << endl;
    // cout << BLOCK_COL_SIZE << " " << BLOCK_ROW_SIZE << " " << BLOCK_PLANE_SIZE << endl; 
    // cout << "grid dimensions: " << dimGrid.z << " " << dimGrid.y << " " << dimGrid.x << endl;
    // cout << "block dimensions: "  << dimBlock.z << " " << dimBlock.y << " " << dimBlock.x << endl; 
    basic_stencil_kernel<<<dimGrid, dimBlock>>>(I_d, O_d, l, m, n); 
    
    cudaMemcpy(O_h, O_d, I_sz, cudaMemcpyDeviceToHost); 

    cudaFree(I_d);
    cudaFree(O_d);
}

int main() {
    int l = 3, m = 3, n = 3; 
    float I[l][m][n] = {{{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}},
                        {{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}},
                        {{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}}};
    float O[l][m][n]; 

    basic_stencil((float*)I, (float*)O, l, m, n); 

    for(int i = 0; i < l; i++) {
        for(int j = 0; j < m; j++) {
            for(int k = 0; k < n; k++) {
                cout << O[i][j][k] << " "; 
            }
            cout << endl; 
        }
        cout << endl; 
    }

}