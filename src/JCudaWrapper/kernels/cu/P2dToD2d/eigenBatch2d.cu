#include <cuda_runtime.h>
#include <math.h>

/**
 * Swap function for double values.
 *
 * @param a First value.
 * @param b Second value.
 */
__device__ inline void swap(double& a, double& b) {
    double temp = a;
    a = b;
    b = temp;
}
class Get{
public:
    const int idx;             ///< Linear index of the element being processed by the current thread.
    const int downSampleFactorXY; ///< Downsampling factor in the x and y dimensions.
    const int layerSize;       ///< Size of a single 2D slice (height * width).
    const int frame;           ///< Index of the current frame.
    const int row;
    const int col;

    /**
     * @brief Constructs a Get object to calculate indices for accessing elements in a 3D data batch.
     * @param inputIdx The linear index of the element being processed by the current thread, before downsampling.
     * @param width The width of each 2D slice.
     * @param depth The number of slices along the depth dimension (per frame).
     * @param downSampleFactorXY The downsampling factor applied in the x and y dimensions.
     */
    __device__ Get(const int inputIdx, const int* dim, const int downSampleFactorXY)
    : idx(inputIdx), 
      downSampleFactorXY(downSampleFactorXY), 
      layerSize(dim[4]), 
      frame(idx / dim[4]),
      row((idx % dim[0]) * downSampleFactorXY),
      col(((idx % dim[4])/dim[0]) * downSampleFactorXY) {}

    

    /**
     * @brief Computes the column-major index within a single 2D slice (height x width).
     * @tparam T The data type of the array elements.
     * @param ld Array of leading dimensions for each 2D slice.
     * @param ldld Leading dimension of the ld array.
     * @return The column-major index within the current 2D slice.
     */
    __device__ int word(const int* ld, const int ldld) const{
        return col * ld[page(ldld)] + row;
    }

    /**
     * @brief Computes the index into the array of pointers (`src`) to access the correct 2D slice.
     * @param ldPtr Leading dimension of the array of pointers.
     * @return The index of the pointer to the current 2D slice.
     */
    __device__ int page(const int ldPtr) const{
        return frame * ldPtr;
    }
    
    
    /**
     * @brief Retrieves a value from the source data array based on the calculated multi-dimensional index.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ T operator()(const T** src, const int* ld, const int ldld, const int ldPtr) {
        return src[page(ldPtr)][word(ld, ldld)];
    }
    
    
    /**
     * @brief Sets the value from the source data array based on the calculated multi-dimensional index.
     * @tparam T The data type of the array elements.
     * @param src Array of pointers, where each pointer points to the beginning of a 2D slice.
     * @param ld Array of leading dimensions for each 2D slice (corresponding to the pointers in src).
     * @param ldld Leading dimension of the ld array (stride between leading dimensions in memory).
     * @param ldPtr Leading dimension of the src array (stride between pointers to different slices in memory).
     * @return The value at the computed index within the specified slice.
     */
     template <typename T>
    __device__ void set(T** src, const int* ld, const int ldld, const int ldPtr, T val) {
        src[page(ldPtr)][word(ld, ldld)] = val;
    }
    
    /**
     * @brief Prints the internal state of the Get object.
     */
    __device__ void print(const int* ld, int ldld, int ldPtr) const {
        printf("Get\n idx: %d, frame: %d: layerSize: %d, downSampleFactorXY: %d, \ncol: %d, row: %d, page: %d, word: %d, ld: %d, ldld: %d, ldPtr: %d\n\n",
               idx, frame, layerSize, downSampleFactorXY, col, row, page(ldPtr), word(ld, ldld), ld[page(ldld)], ldld, ldPtr);
    }
};



/**
 * @class Matrix2x2
 * @brief Represents a 2x2 symmetric matrix.
 */
class Matrix2x2 {
private:
    double mat[2][2];  ///< 2x2 matrix storage.
    double tolerance;  ///< Tolerance for zero checks.

    __device__ double zeroBar(double maybeNear0) {
        return fabs(maybeNear0) <= tolerance ? 0 : maybeNear0;
    }

public:
    __device__ explicit Matrix2x2(const double xx, const double xy, const double yy, double tol) : tolerance(tol) {
        mat[0][0] = zeroBar(xx);
        mat[0][1] = mat[1][0] = zeroBar(xy);
        mat[1][1] = zeroBar(yy);
    }

    __device__ double trace() const { return mat[0][0] + mat[1][1]; }
    __device__ double determinant() const { return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]; }
    __device__ double operator()(int row, int col) const { return mat[row][col]; }
    
    /**
     * @brief Prints the contents of the Matrix2x2 object.
     * For debugging purposes.
     */
    __device__ void print() const {
        printf("Matrix2x2 Debug Info (Thread %d)\n", threadIdx.x + blockIdx.x * blockDim.x);
        printf("  [ %.6f  %.6f ]\n", mat[0][0], mat[0][1]);
        printf("  [ %.6f  %.6f ]\n", mat[1][0], mat[1][1]);
        printf("  Tolerance: %.6f\n", tolerance);
        printf("  Trace: %.6f\n", trace());
        printf("  Determinant: %.6f\n", determinant());
        printf("\n");
    }
};

/**
 * https://people.math.harvard.edu/~knill/teaching/math21b2004/exhibits/2dmatrices/index.html
 * @class Vec
 * @brief A simple wrapper for a double array representing a 2D vector.
 */
class Vec {
private:
    double data[2];
    double tolerance;

    __device__ void sortDescending() {
        if (data[0] < data[1]) swap(data[0], data[1]);
    }

public:
    __device__ Vec(double tol) : tolerance(tol) { data[0] = 0; data[1] = 0; }
    __device__ void set(double x, double y) { data[0] = x; data[1] = y; }
    __device__ double& operator[](int i) { return data[i]; }
    __device__ double operator()(int i) const { return data[i]; }

    __device__ double lengthSquared() const {return data[0] * data[0] + data[1] * data[1];}
    __device__ double length() const { return sqrt(lengthSquared()); }

    __device__ void normalize() {
        double len = length();
        if (len > tolerance) {
            double invLen = 1.0 / len;
            data[0] *= invLen; data[1] *= invLen;
            if (data[1] < 0 || (fabs(data[1]) <= tolerance && data[0] < 0)) {
                data[0] *= -1; data[1] *= -1;
            }
        }
    }

    __device__ void setEigenVec(const Matrix2x2& mat, const double eVal, int vecInd) {
        if(fabs(mat(1, 0)) > tolerance) set(eVal - mat(1, 1), mat(1, 0));
        else if(fabs(mat(0, 1)) > tolerance) set(mat(0, 1), eVal - mat(0, 0));
        else if(vecInd) set(1, 0); 
        else set(0, 1);
        
        normalize();
    }

    __device__ float angle() const { 
        return (lengthSquared() <= 1e-6) ? NAN : atan2(data[1], data[0]); 
    }

    __device__ void setEVal(const Matrix2x2& mat) {
        double t = mat.trace(), d = mat.determinant();
        double x = t/2, y = sqrt(max(t*t/4-d, (double)0));
        data[0] = x + y;
        data[1] = x - y;
    }
    
    __device__ double coherence() const {
        return (data[0] <= tolerance) ? 0 : (float)((data[0] - data[1]) / (data[0] + data[1]));
    }
    
    __device__ void writeTo(float* dst){
        dst[0] = (float)data[0];
        dst[1] = (float)data[1];
    }
    
    /**
     * @brief Prints pertinent debugging information for the Vec object.
     * Includes vector components and results of key methods.
     */
    __device__ void print() const {
        printf("Vec Debug Info (Thread %d)\n", threadIdx.x + blockIdx.x * blockDim.x);
        printf("  Components: [x=%.6f, y=%.6f]\n", data[0], data[1]);
        printf("  Tolerance: %.6f\n", tolerance);
        printf("  Length: %.6f\n", length());
        printf("\n");
    }
};

/**
 * @brief CUDA Kernel to compute eigenvalues/vectors of 2x2 matrices with downsampling.
 *
 * @param n_ds Total number of *downsampled* elements (pixels * frames).
 * @param xx, xy, yy Input structure tensor components (arrays of pointers).
 * @param ldxx, ldxy, ldyy Leading dimensions (heights) for input tensors.
 * @param ldldxx, ... (unused, but kept for consistency with 3D example if needed).
 * @param ldPtrxx, ... (unused, but kept for consistency).
 * @param eVecs, coherence, angle Output arrays (arrays of pointers).
 * @param ldEVec, ldCoh, ldAng Leading dimensions (heights) for output tensors.
 * @param ldldEVec, ... (unused).
 * @param ldPtrEVec, ... (unused).
 * @param dim Original dimensions {height, width, numFrames, imageSize}.
 * @param dsDim Downsampled dimensions {h_ds, w_ds, numFrames, imageSize_ds}.
 * @param downSampleFactor Downsampling factor (e.g., 2, 4).
 * @param eigenInd Index of eigenvector (0 for primary, 1 for secondary).
 * @param tolerance Floating-point tolerance.
 */
extern "C" __global__ void eigenBatch2dKernel(
    const int n_ds, // Use n_ds to indicate it's the downsampled size

    const double** xx, const int* ldxx, const int ldldxx, const int ldPtrxx,
    const double** xy, const int* ldxy, const int ldldxy, const int ldPtrxy,
    const double** yy, const int* ldyy, const int ldldyy, const int ldPtryy,

    float** eVecs, const int* ldEVec, const int ldldEVec, const int ldPtrEVec,
    float** coherence, const int* ldCoh, const int ldldCoh, const int ldPtrCoh,
    float** angle, const int* ldAng, const int ldldAng, const int ldPtrAng,

    const int* dim,

    const int downSampleFactor, // Add downsampling factor
    const int eigenInd,
    const double tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n_ds) return; // Check bounds against downsampled size

    Get src(idx, dim, downSampleFactor);
    
    const Matrix2x2 mat(
    	src(xx, ldxx, ldldxx, ldPtrxx), src(xy, ldxy, ldldxy, ldPtrxy),
                                        src(yy, ldyy, ldldyy, ldPtryy),
        tolerance
    );

    Vec eVals(1e-5);
    eVals.setEVal(mat);
    
    Get dst(idx, dim, 1);

    dst.set(coherence, ldCoh, ldldCoh, ldPtrCoh, (float)eVals.coherence());

    Vec vec(tolerance);
    vec.setEigenVec(mat, eVals(eigenInd), eigenInd);
    
    /*if(idx == 114 + 19*4 + 0) {
        mat.print();
        vec.print();
        dst.print(ldxy, ldldxy, ldPtrxy);
        dst.print(ldEVec, ldldEVec, ldPtrEVec);
    }*/
    
    vec.writeTo(eVecs[dst.page(ldPtrEVec)] + dst.col * ldEVec[dst.page(ldldEVec)] + 2 * dst.row);
    dst.set(angle, ldAng, ldldAng, ldPtrAng, vec.angle());
}
