__device__ int ind(int useableInd, int ld, int height){
	return 3 * useableInd / height * ld + 3 * useableInd % height;
}

//Takes a regular 3d vector with angle between 0 and 2pi and maps it to a nematic vector between 0 and pi.

extern "C" __global__
void vecToNematicKernel(int n, const float* src, const int ldSrc, const int heightSrc, float* dst, const int ldDst, const int heightDst) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    const float* vec = src + ind(idx, ldSrc, heightSrc);
    float* nematic = dst + ind(idx, ldDst, heightDst);

    if(vec[1] < 0 || (vec[1] == 0 && vec[0] < 0)) for(int i = 0; i < 3; i++) nematic[i] = -1 * vec[i];
    else if(dst != src) for(int i = 0; i < 3; i++) nematic[i] = vec[i];
        
}
