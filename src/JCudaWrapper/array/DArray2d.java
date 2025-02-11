package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;

/**
 *
 * @author E. Dov Neimand
 */
public class DArray2d extends Array2d implements DLineArray {

    /**
     * Constructs a sub array of the proffered array.
     *
     * @param src The super array.
     * @param startLine The line in src that this array starts.
     * @param numLines The number of lines in this array.
     * @param startEntry The start index on each included line.
     * @param entriesPerLine The number of entries on each included line.
     */
    public DArray2d(LineArray src, int startLine, int numLines, int startEntry, int entriesPerLine) {
        super(src, startLine, numLines, startEntry, entriesPerLine);
    }

    /**
     * Constructs a 2d array.
     *
     * @param numLines The number of lines in the array.
     * @param entriesPerLine The number of entries on each line of the array.
     */
    public DArray2d(int numLines, int entriesPerLine) {
        super(numLines, entriesPerLine, Sizeof.DOUBLE);
    }

    /**
     * Creates this 2d array from a 1d array.
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line in the new structure.
     */
    public DArray2d(DArray src, int entriesPerLine) {
        super(src, entriesPerLine);
    }
    
    /**
     * Creates this 2d array from a 1d array.
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line in the new structure.
     * @param ld The number of entries between the first element of each line.
     */
    public DArray2d(DArray src, int entriesPerLine, int ld) {
        super(src, entriesPerLine, ld);
    }
    
    

    /**
     * A sub array of the proffered array.
     *
     * @param startLine The line in src that this array starts.
     * @param numLines The number of lines in this array.
     * @param startEntry The start index on each included line.
     * @param entriesPerLine The number of entries on each included line.
     * @return a sub array of the proffered array.
     */
    public DArray2d sub(int startLine, int numLines, int startEntry, int entriesPerLine) {
        return new DArray2d(this, startLine, numLines, startEntry, entriesPerLine);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray2d set(Handle handle, DArray from) {
        super.set(handle, from); 
        return this;
    }

    
    
    /**
     * {@inheritDoc}
     */
    @Override
    public DArray2d copy(Handle handle) {
        return new DArray2d(linesPerLayer(), entriesPerLine())
                .set(handle, this);
    }

    
    /**
     * Performs the matrix-vector multiplication:
     *
     * <pre>
     * this = timesAx * op(A) * X + beta * this
     * </pre>
     *
     * Where op(A) can be A or its transpose.
     *
     * @param handle handle to the cuBLAS library context.
     * @param transA Specifies whether matrix A is transposed (true for
     * transpose and false for not.)
     * @param timesAx Scalar multiplier applied to the matrix-vector product.
     * @param matA Pointer to matrix A in GPU memory.
     * @param vecX Pointer to vector X in GPU memory.
     * @param beta Scalar multiplier applied to vector Y before adding the
     * matrix-vector product.
     * @return this array after this = timesAx * op(A) * X + beta*this
     */
    public DArray addProduct(Handle handle, boolean transA, double timesAx, DArray2d matA, DArray1d vecX, double beta) {

        opCheck(JCublas2.cublasDgemv(
                handle.get(),
                Array.transpose(transA),
                matA.entriesPerLine(), matA.linesPerLayer(),
                P.to(timesAx),
                matA.pointer(), matA.ld(),
                vecX.pointer(), vecX.ld(),
                P.to(beta),
                pointer(),
                ld()
        ));
        return this;
    }
    
    /**
     * Performs the rank-1 update: This is outer product.
     *
     * <pre>
     * this = multProd * X * Y^T + this
     * </pre>
     *
     * Where X is a column vector and Y^T is a row vector.
     *
     * @param handle handle to the cuBLAS library context.
     * @param multProd Scalar applied to the outer product of X and Y^T.
     * @param vecX Pointer to vector X in GPU memory.
     * @param vecY Pointer to vector Y in GPU memory.
     */
    public void outerProd(Handle handle, double multProd, DArray1d vecX, DArray1d vecY) {

        opCheck(JCublas2.cublasDger(
                handle.get(), 
                entriesPerLine(), 
                linesPerLayer(), 
                P.to(multProd), 
                vecX.pointer(), 
                vecX.ld(), 
                vecY.pointer(), 
                vecY.ld(), 
                pointer(), 
                ld()
        ));
    }
    
    
    /**
     * Performs the matrix-matrix multiplication using double precision (Dgemm)
     * on the GPU:
     *
     * <pre>
     * this = op(A) * op(B) + this
     * </pre>
     *
     * Where op(A) and op(B) represent A and B or their transposes based on
     * `transa` and `transb`.
     *
     * @param handle There should be one handle in each thread.
     * @param transposeA True opA should be transpose, false otherwise.
     * @param transposeB True if opB should be transpose, false otherwise.
     * @param timesAB A scalar to be multiplied by AB.
     * @param a Pointer to matrix A, stored in GPU memory. successive rows in
     * memory, usually equal to ARows).
     * @param b Pointer to matrix B, stored in GPU memory.
     * @param timesCurrent This is multiplied by the current array first and
     * foremost. Set to 0 if the current array is meant to be empty, and set to
     * 1 to add the product to the current array as is.
     */
    public void addProduct(Handle handle, boolean transposeA, boolean transposeB, double timesAB, DArray2d a, DArray2d b, double timesCurrent) {

        opCheck(JCublas2.cublasDgemm(handle.get(), // cublas handle
                Array.transpose(transposeA), Array.transpose(transposeB),
                a.entriesPerLine(), b.linesPerLayer(), a.linesPerLayer(),
                P.to(timesAB),
                a.pointer(), a.ld(),
                b.pointer(), b.ld(),
                P.to(timesCurrent),
                pointer(), ld()
        ));
    }
    
    /**
     * Performs matrix addition or subtraction.
     *
     * <p>
     * This function computes C = alpha * A + beta * B, where A, B, and C are
     * matrices.
     * </p>
     *
     * @param handle the cuBLAS context handle
     * @param transA specifies whether matrix A is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param transB specifies whether matrix B is transposed (CUBLAS_OP_N for
     * no transpose, CUBLAS_OP_T for transpose, CUBLAS_OP_C for conjugate
     * transpose)
     * @param alpha scalar used to multiply matrix A
     * @param a pointer to matrix A
     * @param beta scalar used to multiply matrix B
     * @param b pointer to matrix B
     * @return this
     *
     */
    public DArray setSum(Handle handle, boolean transA, boolean transB, double alpha, DArray2d a, double beta, DArray2d b) {

        opCheck(JCublas2.cublasDgeam(handle.get(),
                Array.transpose(transA), Array.transpose(transB),
                entriesPerLine(), linesPerLayer(),
                P.to(alpha), a.pointer(), a.ld(),
                P.to(beta), b.pointer(), b.ld(),
                pointer(), ld()
        ));
        return this;
    }
    
    

    /**
     *
     * Performs symmetric matrix-matrix multiplication using.
     *
     * Computes this = A * A^T + timesThis * this, ensuring C is symmetric.
     *
     * @param handle CUBLAS handle for managing the operation.
     * @param transpose
     * @param fill Specifies which part of the matrix is being used (upper or
     * lower).
     * @param alpha Scalar multiplier for A * A^T.
     * @param a Pointer array to the input matrices.
     * @param timesThis Scalar multiplier for the existing C matrix (usually 0
     * for new computation).
     *
     */
    public void matrixSquared(
            Handle handle,
            boolean transpose,
            int fill, // CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER
            double alpha,
            DArray2d a,
            double timesThis) {

        opCheck(JCublas2.cublasDsyrk(handle.get(),
                fill,
                Array.transpose(transpose),
                entriesPerLine(), linesPerLayer(),
                P.to(alpha), a.pointer(), a.ld(),
                P.to(alpha),
                pointer(), ld()
        ));
    }
    
}
