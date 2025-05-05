package JCudaWrapper.array.Double;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.LineArray;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PSingletonTo2d;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;
import jcuda.runtime.JCuda;

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
    public DArray2d(LineArray src, int startEntry, int entriesPerLine, int startLine, int numLines) {
        super(src, startEntry, entriesPerLine, startLine, numLines);        
    }

    /**
     * Constructs a 2d array.
     *
     * @param numLines The number of lines in the array.
     * @param entriesPerLine The number of entries on each line of the array.
     */
    public DArray2d(int entriesPerLine, int numLines) {
        super(entriesPerLine, numLines, Sizeof.DOUBLE);
    }

    /**
     * Creates this 2d array from a 1d array.
     *
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line in the new
     * structure.
     */
    public DArray2d(DArray src, int entriesPerLine) {
        super(src, entriesPerLine);
    }

    /**
     * Takes a preallocated pointer and gives it an array structure.
     * @param to2d The target of the singleton's pointer.
     */
    public DArray2d(Pointer to2d, int entriesPerLine, int numLines, int pitch) {
        super(to2d, entriesPerLine, numLines, pitch, Sizeof.DOUBLE);
    }
    
    

    /**
     * Creates this 2d array from a 1d array.
     *
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line in the new
     * structure.
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
        return new DArray2d(this, startEntry, entriesPerLine, startLine, numLines);
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
        return new DArray2d(entriesPerLine(), linesPerLayer())
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

    /**
     * {@inheritDoc }
     */
    @Override
    public DArray3d as3d(int linesPerLayer) {
        return new DArray3d(this, entriesPerLine(), linesPerLayer);
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        JCuda.cudaDeviceSynchronize();

        StringBuilder sb = new StringBuilder();
        
        for (int i = 0; i < entriesPerLine(); i++)
            sb.append(entriesAt(i).toString()).append("\n");

        return sb.toString();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray1d entriesAt(int index) {
        return new DArray1d(this, index, linesPerLayer());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] get(Handle handle) {
        double[] cpuArray = new double[size()];
        get(handle, Pointer.to(cpuArray));
        return cpuArray;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DArray2d set(Handle handle, double... srcCPUArray) {
        DLineArray.super.set(handle, srcCPUArray); 
        return this;
    }

    
}
