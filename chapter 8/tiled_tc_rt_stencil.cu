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

// instead of t x t x t shared memory tile (t = input tile dim), we have 3 t x t shared memory tiles
//  at any given iteration. we then iteratively process elements in the z plane
__global__ void tiled_tc_stencil_kernel(float* I, float* O, int l, int m, int n) {
    int planeStart = blockIdx.z * OUT_SIZE_PLANE; 
    int row = blockIdx.y * OUT_SIZE_ROW + threadIdx.y;
    int col = blockIdx.x * OUT_SIZE_COL + threadIdx.x; 

    int inTileRowDim = 2 * STENCIL_ORDER + OUT_SIZE_ROW;
    int inTileColDim = 2 * STENCIL_ORDER + OUT_SIZE_COL; 

    // NOTE: this only works for 3D 7-point stencil
    // Only store curr values in a shared memory tile
    //  - grid values within curr are going to be used by multiple threads
    // can store prev and next in registers
    //  - relevant grid values within next and prev are each only going to be used by one thread only

    float prev, next;
    //__shared__ float prev[2 * STENCIL_ORDER + OUT_SIZE_ROW][2 * STENCIL_ORDER + OUT_SIZE_COL]; 
    __shared__ float curr[2 * STENCIL_ORDER + OUT_SIZE_ROW][2 * STENCIL_ORDER + OUT_SIZE_COL]; 
    //__shared__ float next[2 * STENCIL_ORDER + OUT_SIZE_ROW][2 * STENCIL_ORDER + OUT_SIZE_COL]; 

    if(row >= 0 && row < m && col >= 0 && col < n) {
        // load first batch into prev and curr (next will be loaded within the for loop)
        prev = I[planeStart * m * n + row * n * col]; 
        planeStart++;

        for(int plane = planeStart; plane < planeStart + OUT_SIZE_PLANE; plane++) {
            // load values into next, perform calculations, then set prev = curr, curr = next
            if(row == 1 || row == inTileRowDim - 1 || col == 1 || col == inTileColDim - 1) {
                curr[threadIdx.y][threadIdx.x] = I[plane * m * n + row * n + col];
            } 
            next = I[(plane + 1) * m * n + row * n + col];
            __syncthreads(); 
            

            if(threadIdx.y >= 1 && threadIdx.y < inTileRowDim - 1 && threadIdx.x >= 1 && threadIdx.x < inTileColDim - 1) {                            
                O[plane * m * n + row * n + col] = c0 * curr[threadIdx.y][threadIdx.x]
                                                 + c1 * prev
                                                 + c2 * next
                                                 + c3 * curr[threadIdx.y - 1][threadIdx.x]
                                                 + c4 * curr[threadIdx.y + 1][threadIdx.x]
                                                 + c5 * curr[threadIdx.y][threadIdx.x - 1]
                                                 + c6 * curr[threadIdx.y][threadIdx.x + 1]; 
            }
            
            // write-after-read (false) dependency - need to make sure all the threads have read the current data before updating it
            __syncthreads();
            
            // syncthreads() already called again in next iteration right after updating next -> will automatically sync for prev and curr
            //  as well, so no need to explicitly sync again after these updates
            prev = curr[threadIdx.y][threadIdx.x]; 
            curr[threadIdx.y][threadIdx.x] = next;
        }
    }
}

void tiled_tc_stencil(float* I_h, float* O_h, int l, int m, int n) {
    int size = l * m * n * sizeof(float);
    float *I_d, *O_d;
    
    cudaMalloc((void**)&I_d, size);
    cudaMalloc((void**)&O_d, size);

    cudaMemcpy(I_d, I_h, size, cudaMemcpyHostToDevice); 

    dim3 dimGrid(
        ceil((n - 2 * STENCIL_ORDER) / (float)OUT_SIZE_COL), 
        ceil((m - 2 * STENCIL_ORDER) / (float)OUT_SIZE_ROW), 
        ceil((l - 2 * STENCIL_ORDER) / (float)OUT_SIZE_PLANE)
    );
    dim3 dimBlock(OUT_SIZE_COL + 2 * STENCIL_ORDER, OUT_SIZE_ROW + 2 * STENCIL_ORDER, 1); 
    tiled_tc_stencil_kernel<<<dimGrid, dimBlock>>>(I_d, O_d, l, m, n);

    cudaMemcpy(O_h, O_d, size, cudaMemcpyDeviceToHost); 

    cudaFree(I_d);
    cudaFree(O_d);
}

int main() {
    int l = 4, m = 4, n = 4; 
    
    float I[l][m][n];
    float O[l][m][n]; 
    for(int i = 0; i < l; i++) {
        for(int j = 0; j < m; j++) {
            for(int k = 0; k < n; k++) {
                I[i][j][k] = 1.0; 
                O[i][j][k] = 0.0; 
            }
        }
    }
    tiled_tc_stencil((float*)I, (float*)O, l, m, n); 

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

