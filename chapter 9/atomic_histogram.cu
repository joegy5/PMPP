#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda/atomic>
using namespace std;

#define BLOCK_ROW_SIZE 1
#define BLOCK_COL_SIZE 1
#define NUM_POSSIBLE_VALUES 10

// Atomic operations significantly slow down massively parallel programs
// if two or more threads access the same position in memory at the same time without atomics:
//  - for reads: compiler optimizes by simply reading from that location once, and giving the read item to all the threads accessing it
//  - for writes: compiler cheats by simply choosing one thread at random to actually write to the memory location; the others are silently discarded
//  - with atomics, the reads and writes genuinely are performed by the compiler once for every thread -> much slower, truly serialized
//  - Also, without atomics, two threads can read first, then do internal calculations in parallel, then write back
//      - with atomics, one thread reads, does internal calculations, then writes back, and only THEN can another thread do the same 
//          - this removes the ability for parallel internal calculations across multiple threads
// 
// NOTE: in modern GPUs, atomics have improved a lot by reducing access latency 
//  - instead of reading from DRAM (hundreds of clock cycles), they are read from L2 cache -> orders of magnitude faster
//      - cache miss -> fetch from DRAM, then store in cache for future accesses
//  - modern atomics now almost as good as alternatives (e.g. shared memory, privatization), and also significantly easier to read

__global__ void atomic_histogram_kernel(unsigned char* I, unsigned int* bins, int m, int n, int num_bins) {
    // get the global I position of the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    // get the value at that position, and put into the appropriate bin
    // remember that unsigned char holds only 256 values, enough to hold all possible pixel values from 0-255
    unsigned char pixel_value = I[row * n + col]; 
    // unsigned int = type of object that this atomic op is dealing with
    // thread_scope_device means that this atomic op is applied to all threads within the same device (GPU)
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bin_ref(bins[pixel_value]); 
    // first parameter is the amount to add to the position in memory
    // cuda::std::memory_order_relaxed means no need to impose additional restrictions on how the CUDA compiler can reorder the 
    //  operations around this atomic op. This is because the compiler is already restricted, since the position to perform the atomic op on
    //  (determined by pixel_value) is determined at runtime, meaning it has to wait for that position to be determined first before
    //  performing the atomic op 
    bin_ref.fetch_add(1, cuda::std::memory_order_relaxed); 
    
}

void atomic_histogram(unsigned char* I_h, unsigned int* bins_h, int m, int n, int num_bins, int bin_interval_size) {
    int I_size = m * n * sizeof(unsigned char);
    int bins_size = NUM_POSSIBLE_VALUES * sizeof(unsigned int); 
    unsigned char* I_d;
    unsigned int* bins_d;

    cudaMalloc((void**)&I_d, I_size); 
    cudaMalloc((void**)&bins_d, bins_size);

    cudaMemcpy(I_d, I_h, I_size, cudaMemcpyHostToDevice); 
    
    dim3 dimGrid(ceil(n / (float)BLOCK_COL_SIZE), ceil(m / (float)BLOCK_ROW_SIZE), 1);
    dim3 dimBlock(BLOCK_COL_SIZE, BLOCK_ROW_SIZE, 1);
    atomic_histogram_kernel<<<dimGrid, dimBlock>>>(I_d, bins_d, m, n, num_bins); 

    cudaMemcpy(bins_h, bins_d, bins_size, cudaMemcpyDeviceToHost);

    cudaFree(I_d);
    cudaFree(bins_d);
    
}

int main() {
    int m = 4, n = 4, num_bins = 5;
    int bin_interval_size = ceil(NUM_POSSIBLE_VALUES / (float)num_bins);

    unsigned char I[4][4];
    unsigned int bins[NUM_POSSIBLE_VALUES];

    int curr_value = 0; 
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            I[i][j] = (curr_value++) % NUM_POSSIBLE_VALUES; 
            printf("%d ", I[i][j]); 
        }
        cout << endl; 
    }
    cout << endl; 

    atomic_histogram((unsigned char*)I, (unsigned int*)bins, m, n, num_bins, bin_interval_size);

    for(unsigned int bin : bins) {
        cout << bin << " ";
    }
    cout << endl; 
}

