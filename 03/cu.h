#define __device__
#define __host__
#define __global__

struct P {
    int x;
    int y;
    int z;
};


struct P blockIdx;
struct P blockDim;
struct P threadIdx;
