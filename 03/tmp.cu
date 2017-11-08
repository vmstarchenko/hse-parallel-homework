#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

static const int BLOCK_SIZE = 256;
static const int N = 2000;

__global__ void vadd(int* a, int* b, int* c, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d\n", id);
    if (id < N) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    // host (h*) and device (d*) pointers
    int *ha, *hb, *hc, *da, *db, *dc;
    int i;

    // allocate host memory
    ha = new int[N];
    hb = new int[N];
    hc = new int[N];

    // allocate device memory
    cudaMalloc((void**)&da, N * sizeof(int));
    cudaMalloc((void**)&db, N * sizeof(int));
    cudaMalloc((void**)&dc, N * sizeof(int));

    // initialize input vectors
    for (i = 0; i < N; i++) {
        ha[i] = rand() % 10000;
        hb[i] = rand() % 10000;
    }

    // copy input vectors to device
    cudaMemcpy(da, ha, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, sizeof(int) * N, cudaMemcpyHostToDevice);

    // run kernel
    int grid = ceil(N * 1.0 / BLOCK_SIZE);
    vadd<<<grid, BLOCK_SIZE>>>(da, db, dc, N);
    cudaDeviceSynchronize();

    // copy output vector to host
    cudaMemcpy(hc, dc, sizeof(int) * N, cudaMemcpyDeviceToHost);

    // check results
    for (i = 0; i < N; i++) {
        if (hc[i] != ha[i] + hb[i]) {
            printf("Error at index %i : %i != %i\n", i, hc[i], ha[i] + hb[i]);
        }
    }

    // free GPU memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    // free CPU memory
    delete[] ha;
    delete[] hb;
    delete[] hc;

    // clean up device resources
    cudaDeviceReset();

    return 0;
}
