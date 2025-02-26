
extern "C" __global__ void addEBEQuotientKernel(
    const int n,
    float* dst,
    const int dstInc,
    const float timesQuotient,
    const float* numerator, 
    const int numerInc, 
    const float* denom, 
    const int denomInc,
    const float timesThis
        
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    int toInd = idx*dstInc;
    dst[toInd] = timesThis * dst[toInd] + timesQuotient * numerator[idx * numerInc] / denom[idx * denomInc];
}

