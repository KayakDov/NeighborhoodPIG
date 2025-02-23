//Takes a regular 3d vector with angle between 0 and 2pi and maps it to a nematic vector between 0 and pi.

extern "C" __global__
void vecToNematicKernel(int n, const double* src, int ldSrc, double* dst, int ldDst) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    const double* vec = src + idx * ldSrc;
    double* nematic = dst + idx * ldDst;

    if(vec[1] < 0 || (vec[1] == 0 && vec[0] < 0)) for(int i = 0; i < 3; i++) nematic[i] = -1 * vec[i];
    else if(dst != src) for(int i = 0; i < 3; i++) nematic[i] = vec[i];
        
}
