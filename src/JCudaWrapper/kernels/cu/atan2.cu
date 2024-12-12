
extern "C" __global__
void atan2Kernel(int n, const double* vectors, int ldFrom, double* angles, int incTo) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;
    
    const double* vecFrom = vectors + idx*ldFrom;
    double* vecTo = angles + idx*incTo;
    
    *vecTo = atan2(vecFrom[1], vecFrom[0]);    
}
