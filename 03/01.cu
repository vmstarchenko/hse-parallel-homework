#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include <sys/time.h>

#include <cuda.h>

static const int BLOCK_SIZE = 256;

double EPS = 0.001;

void free_matrix(double* matrix);
void print_matrix(int N, int M, double* matrix);
void print_matrix_1d(int N, double* matrix);
bool allclose(int n, double eps, double* m1, double* m2);
double get_time_diff(struct timeval& start, struct timeval& end);

double random_double() {
    return rand() / (RAND_MAX + 1.);
}

double* generate_matrix(int N, int M) {
    double* matrix_1d = (double*)malloc(N * M * sizeof(*matrix_1d));

    for (int i = 0, p = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j, ++p) {
            matrix_1d[p] = random_double();
        }
    }

    return matrix_1d;
}

__host__ void transpose_cpu(int N, int M, double* src, double* dst) {

    int NM = N * M;

    for (int p = 0; p < NM; ++p) {
        dst[(p % M) * N + (p / M)] = src[p];
    }
}

__global__ void transpose_gpu(int N, int M, double* src, double* dst) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;

    if (p < N * M) {
        dst[(p % M) * N + (p / M)] = src[p];
    }
}

// m1: MxN; m2 MxK; res: NxK; note: m1 transposed
__host__ void mult_cpu(int N, int M, int K, double* m1, double* m2,
                       double* res) {
    int NK = N * K;
    for (int p = 0; p < NK; ++p) {
        int i = p / K, j = p % K;
        res[p] = 0;
        for (int m = 0; m < M; ++m) {
            res[p] += m1[m * N + i] * m2[m * K + j];
        }
    }
}

__global__ void mult_gpu(int N, int M, int K, double* m1, double* m2,
                         double* res) {
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    int NK = N * K;

    if (p < NK) {
        int i = p / K, j = p % K;
        printf("p: %d %d %d\n", p, i, j);
        double res_p = 0;
        for (int m = 0; m < M; ++m) {
            res_p += m1[m * N + i] * m2[m * K + j];
        }
        res[p] = res_p;
    }
}

int main() {
    // int best_grid, best_block;
    // cudaOccupancyMaxPotentialBlockSize(&best_grid, &best_block,
    // transpose_gpu);
    // printf("Best grid size: %d; best block size: %d\n\n", best_grid,
    //        best_block);
    // return 0;

    int N = 2; // 100;
    int M = 3; // 110;
    int K = 3; // 120;
    int NM = N * M;
    int MK = M * K;
    int NK = N * K;

    struct timeval start, end;
    cudaEvent_t startT, endT;
    int global_iters = 100;
    int iters = global_iters;
    double summ_time = 0;
    float cur_time = 0;
    cudaStream_t str;

    double* mNM_h = generate_matrix(N, M);
    double* mMN_h = generate_matrix(M, N);
    double* mMK_h = generate_matrix(M, K);
    double* mNK_h = generate_matrix(N, K);

    double* mNM_d;
    double* mMN_d;
    double* mMK_d;
    double* mNK_d;

    gettimeofday(&start, NULL);
    cudaMalloc((void**)&mNM_d, sizeof(*mNM_d) * NM);
    cudaMalloc((void**)&mMN_d, sizeof(*mMN_d) * NM);
    cudaMalloc((void**)&mMK_d, sizeof(*mMK_d) * MK);
    cudaMalloc((void**)&mNK_d, sizeof(*mNK_d) * NK);
    cudaMemcpy(mNM_d, mNM_h, sizeof(*mNM_h) * NM, cudaMemcpyHostToDevice);
    cudaMemcpy(mMK_d, mMK_h, sizeof(*mMK_h) * NM, cudaMemcpyHostToDevice);
    gettimeofday(&end, NULL);
    printf("gpu malloc and memcopy time: %lf\n", get_time_diff(start, end));

    if (NM < 50) {
        print_matrix(N, M, mNM_h);
        printf("\n");
        print_matrix(M, K, mMK_h);
        printf("\n");
    }

    /// Transpose //////////////////////////////////////////////////////////////
    // begin transpose_cpu
    summ_time = 0;
    iters = global_iters / 10;
    for (int i = 0; i < iters; ++i) {
        gettimeofday(&start, NULL);
        transpose_cpu(N, M, mNM_h, mMN_h);
        gettimeofday(&end, NULL);
        summ_time += get_time_diff(start, end);
    }

    double cpu_transpose_time = summ_time / iters;
    printf("transpose by cpu time %lf\n", cpu_transpose_time);
    // end transpose_cpu

    // begin transpose_gpu
    cudaEventCreate(&startT);
    cudaEventCreate(&endT);
    cudaStreamCreate(&str);

    dim3 threads(32, 32);

    int grid = ceil((NM * 1.0) / BLOCK_SIZE);

    summ_time = 0;
    cur_time = 0;
    iters = global_iters;
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(startT, str);

        transpose_gpu<<<grid, BLOCK_SIZE>>>(N, M, mNM_d, mMN_d);

        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&cur_time, startT, endT);

        summ_time += ((double)cur_time) / 1000;
    }
    double gpu_transpose_time = summ_time / iters;
    printf("transpose by gpu time %lf\n", gpu_transpose_time);

    // transpouse conclusion
    if (allclose(M * N, EPS, mMN_h, mMN_d)) {
        printf("OK: same results\n");
    } else {
        printf("ERROR: different results\n");
    }
    printf("transpose speedup: %f\n", cpu_transpose_time / gpu_transpose_time);
    // TODO: matrix the same
    // end transpose_gpu

    ////////////////////////////////////////////////////////////////////////////

    /// Multiplication /////////////////////////////////////////////////////////
    // begin mult_cpu
    summ_time = 0;
    iters = global_iters;
    for (int i = 0; i < iters; ++i) {
        gettimeofday(&start, NULL);
        mult_cpu(N, M, K, mMN_h, mMK_h, mNK_h);
        gettimeofday(&end, NULL);
        summ_time += get_time_diff(start, end);
    }
    double cpu_mult_time = summ_time / iters;
    printf("mult by cpu time: %lf\n", cpu_mult_time);

    // print_matrix(N, K, mNK_h);
    // end mult_cpu

    // begin mult_gpu
    cudaEventCreate(&startT);
    cudaEventCreate(&endT);
    cudaStreamCreate(&str);

    grid = ceil((M * 1.0) / BLOCK_SIZE);

    summ_time = 0;
    cur_time = 0;
    iters = 1; // TODO
    for (int i = 0; i < iters; ++i) {
        cudaEventRecord(startT, str);

        mult_gpu<<<grid, BLOCK_SIZE>>>(N, M, K, mMN_d, mMK_d, mNK_d);

        cudaEventRecord(endT, str);
        cudaEventSynchronize(endT);
        cudaEventElapsedTime(&cur_time, startT, endT);

        summ_time += ((double)cur_time) / 1000;
    }
    double gpu_mult_time = summ_time / iters;
    printf("mult by gpu time %lf\n", gpu_mult_time);

    // mult conclusion
    if (allclose(N * K, EPS, mNK_h, mNK_d)) {
        printf("OK: same results\n");
    } else {
        printf("ERROR: different results\n");
    }
    printf("transpose speedup: %f\n", cpu_transpose_time / gpu_transpose_time);
    // end mult_gpu

    cudaFree(mNM_d);
    cudaFree(mMN_d);
    free_matrix(mNM_h);
    free_matrix(mMN_h);
    free_matrix(mMK_h);

    return 0;
}

void print_matrix(int N, int M, double* matrix) {
    for (int i = 0, p = 0; i < N; ++i) {
        printf("%lf", matrix[p++]);
        for (int j = 1; j < M; ++j, ++p) {
            printf(" %lf", matrix[p]);
        }
        printf("\n");
    }
}

void print_matrix_1d(int N, double* matrix) {
    printf("%lf", matrix[0]);
    for (int i = 1; i < N; ++i) {
        printf(" %lf", matrix[i]);
    }
    printf("\n");
}

void free_matrix(double* matrix) {
    free(matrix);
}

bool allclose(int n, double eps, double* m_h, double* m_d) {
    double* md_h = (double*)malloc(sizeof(*md_h) * n);
    cudaMemcpy(md_h, m_d, sizeof(*md_h) * n, cudaMemcpyDeviceToHost);
    double diff = 0;
    double neps = -eps;

    if (n < 50) {
        printf("Host matrix:\n");
        print_matrix_1d(n, m_h);
        printf("Device matrix:\n");
        print_matrix_1d(n, md_h);
        printf("\n");
    }

    for (int i = 0; i < n; ++i) {
        diff = m_h[i] - md_h[i];
        if (diff > eps || diff < neps) {
            return false;
        }
    }

    free(md_h);

    return true;
}

double get_time_diff(struct timeval& start, struct timeval& end) {
    return ((double)((end.tv_sec * 1000000 + end.tv_usec) -
                     (start.tv_sec * 1000000 + start.tv_usec))) /
           1000000;
}
