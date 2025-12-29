
extern "C" __global__ void VisualizeVectorsKernel(
	const int n, 
	const float *vectors, const int ldVecs, 
	float *to, int ldTo,
	float* coherence, int ldCoherence,
	const int height, const int width, int vecFreq
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx >= n) return;
}
