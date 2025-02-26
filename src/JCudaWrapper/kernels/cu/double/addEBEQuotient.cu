
extern "C" __global__ void addEBEQuotientKernel(
    const int n,
    double* dst,
    const int dstInc,
    const double timesQuotient,
    const double* numerator, 
    const int numerInc, 
    const double* denom, 
    const int denomInc,
    const double timesThis
        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int toInd = idx*dstInc;
    dst[toInd] = timesThis * dst[toInd] + timesQuotient * numerator[idx * numerInc] / denom[idx * denomInc];
}

