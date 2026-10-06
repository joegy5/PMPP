#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#define FILTER_RADIUS 5
#define out_tile_sz_row 10
#define out_tile_sz_col 10
__constant__ float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// SHARED MEMORY OPTIMIZATION:
//  - load ALL elements that ALL threads in the block need to compute, into shared memory (referred to as input tile)
//  - this is a common pattern in shared memory tiling, applied to tiled matmul as well
//      - much better than loading elements that are only shared by ALL threads, as some threads share elements that they don't share with other threads

// COMMON PATTERN: load more threads than output elements in the block, disable extra threads when needed
//  - number of threads doesn't always have to be the size of the output tile
//  - if the number of elements needed to calculate the output tile is greater than the number of elements in the output tile, consider using this strategy

__global__ void tiled_conv_kernel(float* I, float* O, int m, int n) {
    // Use output tile size, NOT blockDim, since blockDim is TOTAL number of threads in current block (including those not corresponding to output elements); 
    //  if we skip over all of them, we will have skipped over too many elements, since threads from different blocks OVERLAP, unlike in previous kernels
    int row = blockIdx.y * out_tile_sz_row + threadIdx.y - FILTER_RADIUS;
    int col = blockIdx.x * out_tile_sz_col + threadIdx.x - FILTER_RADIUS;
    int filter_dim = 2 * FILTER_RADIUS + 1;
    float res = 0.0;

    __shared__ float in_tile[2 * FILTER_RADIUS + 1 + out_tile_sz_row - 1][2 * FILTER_RADIUS + 1 + out_tile_sz_col - 1]; 
    if(row < 0 || row >= m || col < 0 || col >= n) {
        in_tile[threadIdx.y][threadIdx.x] = 0.0f; 
    } 
    else {
        in_tile[threadIdx.y][threadIdx.x] = I[row * n + col]; 
    }

    __syncthreads(); 
    

    if(threadIdx.y >= FILTER_RADIUS && threadIdx.y < FILTER_RADIUS + out_tile_sz_row && threadIdx.x >= FILTER_RADIUS && threadIdx.x < FILTER_RADIUS + out_tile_sz_col) {
        for(int i = 0; i < filter_dim; i++) {
            for(int j = 0; j < filter_dim; j++) {
                int i_coord = i + threadIdx.y - FILTER_RADIUS; 
                int j_coord = j + threadIdx.x - FILTER_RADIUS; 
                res += in_tile[i_coord][j_coord] * F[i][j]; 
            }
        }
        O[row * n + col] = res;
    }
}



void tiled_conv(float* I_h, float* O_h, float* F_h, int m, int n) {
    int I_sz = m * n * sizeof(float);
    int F_dim = 2 * FILTER_RADIUS + 1;
    int F_sz = F_dim * F_dim;
    float *I_d, *O_d;

    cudaMalloc((void**)&I_d, I_sz); 
    cudaMalloc((void**)&O_d, I_sz);

    cudaMemcpy(I_d, I_h, I_sz, cudaMemcpyHostToDevice);
    // filter never changes, can write to constant memory
    // constant memory never written to -> compiler aggressively caches it instead of leaving in DRAM
    // constant memory doesn't require any write support (can only read from it) -> caching can be made much more efficient, even into specialized constant caches
    //  that constant memory elements can fit into
    // since filter is usually small, it most likely will entirely fit into constant memory -> likely don't need to access DRAM at all
    cudaMemcpyToSymbol(F, F_h, F_sz * sizeof(float)); // copy F_h to constant memory F

    // launch the kernel grid
    // number of threads per block = size of INPUT tile, not output tile -> have each thread load a single element, then disable threads outside the output tile range
    //  - input tile dim = filter dim + block dim = (FILTER_RADIUS + block_sz_row) x (FILTER_RADIUS + block_sz_col)
    // number of actual blocks still stays the same, as each block is meant to compute a (block_sz_row x block_sz_col) part of the output matrix
    // 
    dim3 dimGrid(ceil(n / (float)out_tile_sz_col), ceil(m / (float)out_tile_sz_row));
    dim3 dimBlock(out_tile_sz_col + F_dim - 1, out_tile_sz_row + F_dim - 1, 1); 
    tiled_conv_kernel<<<dimGrid, dimBlock>>>(I_d, O_d, m, n); 

    cudaMemcpy(O_h, O_d, I_sz, cudaMemcpyDeviceToHost); 

    cudaFree(I_d); 
    cudaFree(O_d); 
}

int main() {
    // create the input arrays, as well as the output array
    int num_rows = 1 << 5, num_cols = 1 << 5; 
    float* I = (float*)malloc(num_rows * num_cols * sizeof(float)); 
    float* O = (float*)malloc(num_rows * num_cols * sizeof(float)); 
    float* F = (float*)malloc((2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * sizeof(float));

    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            I[i * num_cols + j] = 1; 
        }
    }
    for(int i = 0; i < 2 * FILTER_RADIUS + 1; i++) {
        for(int j = 0; j < 2 * FILTER_RADIUS + 1; j++) {
            F[i * (2 * FILTER_RADIUS + 1) + j] = 1; 
        }
    }

    // call the kernel function
    tiled_conv((float*)I, (float*)O, (float*)F, num_rows, num_cols);


    for(int i = 0; i < num_rows; i++) {
        for(int j = 0; j < num_cols; j++) {
            cout << O[i * num_cols + j] << " "; 
        }
        cout << endl; 
    }
}








