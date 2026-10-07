#include <stdlib.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#define STENCIL_ORDER 1

#define OUT_SIZE_PLANE 1
#define OUT_SIZE_ROW 1
#define OUT_SIZE_COL 1 

#define c0 1
#define c1 1
#define c2 1
#define c3 1
#define c4 1
#define c5 1
#define c6 1

// Same concept as tiled convolutions. Load ALL the input elements that ALL threads in current block will be using to calculate 
//  output, into shared memory

__global__ void tiled_stencil_kernel(float* I, float* O, int l, int m, int n) {
    // get position of the current thread
    // NOTE: think of thread.z, thread.y, thread.x as absolute positions within a single block, 
    //  and adding blockIdx * OUT_SIZE/blockDim simply shifts all those positions to the right block
    //  -> threadIdx represents absolute positions within a block, blockIdx * blockDim + threadIdx represents absolute positions
    //      within the original input matrix
    // remember to subtract stencil order to shift threads such that the output grid threads are positioned correctly
    
    int plane = blockIdx.z * OUT_SIZE_PLANE + threadIdx.z;
    int row = blockIdx.y * OUT_SIZE_ROW + threadIdx.y;
    int col = blockIdx.x * OUT_SIZE_COL + threadIdx.x;
    
    int inTilePlaneDim = 2 * STENCIL_ORDER + OUT_SIZE_PLANE; 
    int inTileRowDim = 2 * STENCIL_ORDER + OUT_SIZE_ROW;
    int inTileColDim = 2 * STENCIL_ORDER + OUT_SIZE_COL;

    __shared__ float in_tile[2 * STENCIL_ORDER + OUT_SIZE_PLANE][2 * STENCIL_ORDER + OUT_SIZE_ROW][2 * STENCIL_ORDER + OUT_SIZE_COL];

    // load value at this position into shared memory only if this position is within I bounds
    if(plane >= 0 && plane < l && row >= 0 && row < m && col >= 0 && col < n) {
        if(blockIdx.z == 0 && blockIdx.y == 1 && blockIdx.x == 0) {
            printf("thread coord (%d, %d, %d), block coord (%d, %d, %d), global coord (%d, %d, %d), value: %f\n", threadIdx.z, threadIdx.y, threadIdx.x, blockIdx.z, blockIdx.y, blockIdx.x, plane, row, col, I[plane * m * n + row * n + col]);
        }
        in_tile[threadIdx.z][threadIdx.y][threadIdx.x] = I[plane * m * n + row * n + col];
    }
    __syncthreads(); // read-after-write dependency

    // // check if the thread corresponds to one of the actual output grid values we are trying to calculate for this block
    if(threadIdx.z >= 1 && threadIdx.z < inTilePlaneDim - 1 && threadIdx.y >= 1 && threadIdx.y < inTileRowDim - 1 && threadIdx.x >= 1 && threadIdx.x < inTileColDim - 1) {
        O[plane * m * n + row * n + col] = c0 * in_tile[threadIdx.z][threadIdx.y][threadIdx.x]
                                         + c1 * in_tile[threadIdx.z - 1][threadIdx.y][threadIdx.x]
                                         + c2 * in_tile[threadIdx.z + 1][threadIdx.y][threadIdx.x]
                                         + c3 * in_tile[threadIdx.z][threadIdx.y - 1][threadIdx.x]
                                         + c4 * in_tile[threadIdx.z][threadIdx.y + 1][threadIdx.x]
                                         + c5 * in_tile[threadIdx.z][threadIdx.y][threadIdx.x - 1]
                                         + c6 * in_tile[threadIdx.z][threadIdx.y][threadIdx.x + 1]; 
    }
}

void tiled_stencil(float* I_h, float* O_h, int l, int m, int n) {
    int size = l * m * n * sizeof(float); 
    float *I_d, *O_d;
    int stencilDim = 2 * STENCIL_ORDER + 1; 

    cudaMalloc((void**)&I_d, size);
    cudaMalloc((void**)&O_d, size);

    cudaMemcpy(I_d, I_h, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((n - 2) / (float)OUT_SIZE_COL), ceil((m - 2) / (float)OUT_SIZE_ROW), ceil((l - 2) / (float)OUT_SIZE_PLANE));
    dim3 dimBlock(OUT_SIZE_COL + stencilDim - 1, OUT_SIZE_ROW + stencilDim - 1, OUT_SIZE_PLANE + stencilDim - 1);

    tiled_stencil_kernel<<<dimGrid, dimBlock>>>(I_d, O_d, l, m, n); 

    cudaMemcpy(O_h, O_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(I_d);
    cudaFree(O_d); 
}

int main() {
    int l = 4, m = 4, n = 4; 
    float I[l][m][n] = {{{1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}},
                        {{1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}},
                        {{1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}},
                        {{1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}, {1., 1., 1., 1.}}};
    float O[l][m][n]; 

    tiled_stencil((float*)I, (float*)O, l, m, n); 

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


