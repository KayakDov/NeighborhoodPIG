//Takes a regular vector with angle between 0 and 2pi and maps it to a nematic vector between 0 and pi.

extern "C" __global__
void vecToNematicKernel(const double* vectors, int ldVectors, double* nematics, int ldNematics, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= n) return;
    
    const double* vec = vectors + idx * ldVectors;
    double* nematic = nematics + idx * ldNematics;

    if(vec[1] < 0 || (vec[1] == 0 && vec[0] < 0)){
        nematic[0] = -1 * vec[0];
        nematic[1] = -1 * vec[1];
    } else {
        nematic[0] = vec[0];
        nematic[1] = vec[1];
    }
        
}
