package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import java.util.Arrays;
import org.apache.commons.math3.exception.DimensionMismatchException;
import JCudaWrapper.resourceManagement.Handle;

/**
 * This class provides methods for handling a batch of strided matrices stored
 * in row-major format. Each matrix in the batch can be accessed and operated on
 * individually or as part of the batch. The class extends {@code Matrix} and
 * supports operations such as matrix multiplication and addition. The strided
 * storage is designed to support JCuda cuSolver methods that require row-major
 * memory layout.
 *
 * Strided matrices are stored with a defined distance (stride) between the
 * first elements of each matrix in the batch.
 *
 * @author E. Dov Neimand
 */
public class MatricesStride extends Matrix {

    private int subWidth;
    private DStrideArray batchArray;

    /**
     * Constructor for creating a batch of strided matrices. Each matrix is
     * stored with a specified stride and the batch is represented as a single
     * contiguous block in memory.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param subWidth The number of columns (width) in each submatrix.
     * @param stride The number of elements between the first elements of
     * consecutive submatrices in the batch.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int subHeight, int subWidth, int stride, int batchSize) {
        super(handle, subHeight, subWidth * batchSize);
        batchArray = data.getAsBatch(stride, batchSize, subHeight * subWidth);
        this.subWidth = subWidth;
    }

    /**
     * Creates a simple batch matrix with coldDist = height.
     *
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix.
     * @param colDist The distance between the first element of each column.
     */
    public MatricesStride(Handle handle, DStrideArray data, int height, int colDist) {
        this(handle, data, height, data.length / colDist, colDist);
    }

    /**
     * Creates a simple batch matrix with coldDist = height.
     *
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix.
     * @param subWidth The width of each sub matrix.
     * @param colDist The distance between the first element of each column.
     */
    public MatricesStride(Handle handle, DStrideArray data, int height, int subWidth, int colDist) {
        super(handle, data, height, subWidth * data.batchSize, colDist);
        this.batchArray = data;
        this.subWidth = subWidth;
    }

    /**
     * Creates a simple batch matrix with coldDist = height.
     *
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix.
     */
    public MatricesStride(Handle handle, DStrideArray data, int height) {
        this(handle, data, height, height);
    }

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int subHeight, int batchSize) {
        this(handle, subHeight, subHeight, batchSize);
    }

    /**
     * Constructor for creating a batch of square strided matrices. Each matrix
     * is stored with a specified stride and the batch is represented as a
     * single contiguous block in memory. These matrices have no overlap.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param subHeight The number of rows (height) in each submatrix.
     * @param subWidth The number of columns (width) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int subHeight, int subWidth, int batchSize) {
        this(handle, subHeight, subWidth, subHeight * subWidth, batchSize);
    }

    /**
     * Returns a vector of elements corresponding to row {@code i} and column
     * {@code j} across all matrices in the batch. Each element in the vector
     * corresponds to the element at position (i, j) in a different submatrix.
     *
     * @param i The row index in each submatrix.
     * @param j The column index in each submatrix.
     * @return A vector containing the elements at (i, j) for each submatrix in
     * the batch.
     */
    public Vector get(int i, int j) {
        return asVector().getSubVector(index(i, j), batchArray.batchCount(), batchArray.stride);
    }

    /**
     * Retrieves all elements from each submatrix in the batch as a 2D array of
     * {@code Vector} objects. Each vector contains elements corresponding to
     * the same row and column across all matrices in the batch. The method
     * returns a 2D array where each element (i, j) is a vector of the elements
     * at position (i, j) from each matrix in the batch. "i" is the row and "j"
     * is the column.
     *
     * @return A 2D array of {@code Vector} objects. Each {@code Vector[i][j]}
     * represents the elements at row {@code i} and column {@code j} for all
     * submatrices in the batch.
     */
    public Vector[][] parition() {
        Vector[][] all = new Vector[getSubHeight()][getSubHeight()];
        for (int i = 0; i < getSubHeight(); i++) {
            int row = i;
            Arrays.setAll(all[row], col -> get(row, col));
        }
        return all;
    }

    /**
     * Returns the number of columns (width) in each submatrix.
     *
     * @return The width of each submatrix in the batch.
     */
    public int getSubWidth() {
        return subWidth;
    }

    /**
     * Returns the number of rows (height) in each submatrix.
     *
     * @return The height of each submatrix in the batch.
     */
    public int getSubHeight() {
        return getHeight();
    }

    /**
     * Performs matrix multiplication and on the batches of matrices, and add
     * them to this matrix. This method multiplies matrix batches {@code a} and
     * {@code b}, scales the result by {@code timesAB}, scales the existing
     * matrix in the current instance by {@code timesResult}, and then adds them
     * together and palces the result here.
     *
     * @param transposeA Whether to transpose the matrices in {@code a}.
     * @param transposeB Whether to transpose the matrices in {@code b}.
     * @param a The left-hand matrix batch in the multiplication.
     * @param b The right-hand matrix batch in the multiplication.
     * @param timesAB The scaling factor applied to the matrix product
     * {@code a * b}.
     * @param timesResult The scaling factor applied to the result matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatricesStride multAndAdd(boolean transposeA, boolean transposeB, MatricesStride a, MatricesStride b, double timesAB, double timesResult) {
        if (a.getSubWidth() != b.getSubHeight()) {
            throw new DimensionMismatchException(a.getSubWidth(),
                    b.getSubHeight());
        }

        // Perform batched matrix multiplication and addition
        batchArray.multMatMatStridedBatched(getHandle(), transposeA, transposeB,
                transposeA?a.getSubWidth():a.getSubHeight(), 
                transposeA?a.getSubHeight():a.getSubWidth(), 
                transposeB?b.getSubHeight():b.getSubWidth(),
                timesAB,
                a.batchArray, a.colDist,
                b.batchArray, b.colDist,
                timesResult, colDist
        );

        return this;
    }

    /**
     * Computes the eigenvalues. This batch must be a set of symmetric 2x2
     * matrices.
     *
     * @param workSpace Should be at least as long as batchSize.
     * @return The eigenvalues.
     */
    public VectorsStride computeVals2x2(Vector workSpace) {

        if (getHeight() != 2)
            throw new IllegalArgumentException("compute vals 2x2 can only be called on a 2x2 matrix.  These matrices are " + getHeight() + "x" + getSubWidth());

        VectorsStride vals = new VectorsStride(handle, 2, getBatchSize(), 2, 1);
        Vector[] val = vals.parition();

        Vector[][] m = parition();

        Vector trace = workSpace.getSubVector(0, batchArray.batchCount());

        trace.set(m[1][1]);
        trace.addToMe(1, m[0][0]);//= a + d

        val[0].mapEbeMultiplyToSelf(trace, trace); //= (d + a)*(d + a)

        val[1].mapEbeMultiplyToSelf(m[0][1], m[0][1]); //c^2
        val[1].mapAddEbeMultiplyToSelf(m[0][0], m[1][1], -1);// = ad - c^2

        val[0].addToMe(-4, val[1]);//=(d + a)^2 - 4(ad - c^2)
        KernelManager.get("sqrt").mapToSelf(getHandle(), val[0]);//sqrt((d + a)^2 - 4(ad - c^2))

        val[1].set(trace);
        val[1].addToMe(-1, val[0]);
        val[0].addToMe(1, trace);

        vals.mapMultiplyToSelf(0.5);

        return vals;
    }

    /**
     * Computes the eigenvalues for a set of symmetric 3x3 matrices. If this
     * batch is not such a set then this method should not be called.
     *
     * @param workSpace Should have length equal to the width of this matrix.
     * @return The eigenvalues.
     *
     */
    //m := a, d, g, d, e, h, g, h, i = m00, m10, m20, m01, m11, m21, m02, m12, m22
    //p := tr m = a + e + i
    //q := (p^2 - norm(m)^2)/2 where norm = a^2 + d^2 + g^2 + d^2 + ...
    // solve: lambda^3 - p lambda^2 + q lambda - det m        
    public VectorsStride computeVals3x3(Vector workSpace) {

        if (getHeight() != 3)
            throw new IllegalArgumentException("computeVals3x3 can only be called on a 3x3 matrix.  These matrices are " + getHeight() + "x" + getSubWidth());

        VectorsStride vals = new VectorsStride(handle, getHeight(), getBatchSize(), getHeight(), 1);
        Vector[] work = workSpace.parition(3);

        Vector[][] m = parition();

        Vector[][] minor = new Vector[3][3];

        Vector negTrace = negativeTrace(work[0]);//val[0] is taken, but val[1] is free.

        setDiagonalMinors(minor, m, vals);
        Vector C = work[1].fill(0);
        for (int i = 0; i < 3; i++) {
            C.addToMe(1, minor[i][i]);
        }

        setRow0Minors(minor, m, vals);
        Vector det = work[2];
        det.mapEbeMultiplyToSelf(m[0][0], minor[0][0]);
        det.addEbeMultiplyToSelf(-1, m[0][1], minor[0][1], 1);
        det.addEbeMultiplyToSelf(-1, m[0][2], minor[0][2], -1);

//        System.out.println("algebra.Eigen.computeVals3x3() coeficiants: " + trace + ", " + C + ", " + det);
        cubicRoots(negTrace, C, det, vals); // Helper function

        return vals;
    }

    private static DArray negOnes3;

    /**
     * A vector of 3 negative ones.
     *
     * @return A vector of 3 negative ones.
     */
    private static VectorsStride negOnes3(Handle handle, int batchSize) {
        if (negOnes3 == null) negOnes3 = DArray.empty(3).fill(handle, -1, 1);
        return new VectorsStride(handle, negOnes3.getAsBatch(0, batchSize, 3), 1);
    }

    /**
     * The negative of the trace of the submatrices.
     *
     * @param traceStorage The vector that gets overwritten with the trace.
     * Should have batch elements.
     * @param ones a vector that will have -1's stored in it. It should have
     * height number of elements in it.
     * @return The trace.
     */
    private Vector negativeTrace(Vector traceStorage) {//TODO:Retest!

        VectorsStride diagnols = new VectorsStride(
                handle,
                dArray().getAsBatch(9, traceStorage.getDimension(), 9),
                4
        );

        return traceStorage.setBatchVecVecMult(
                diagnols,
                negOnes3(handle, batchArray.batchSize)
        );

    }

    /**
     * Sets the minors of the diagonal elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setDiagonalMinors(Vector[][] minor, Vector[][] m, VectorsStride minorStorage) {

        Vector[] storagePartition = minorStorage.parition();

        for (int i = 0; i < minor.length; i++)
            minor[i][i] = storagePartition[i];

        minor[0][0].mapEbeMultiplyToSelf(m[1][1], m[2][2]);
        minor[0][0].addEbeMultiplyToSelf(-1, m[1][2], m[1][2], 1);

        minor[1][1].mapEbeMultiplyToSelf(m[0][0], m[2][2]);
        minor[1][1].addEbeMultiplyToSelf(-1, m[0][2], m[0][2], 1);

        minor[2][2].mapEbeMultiplyToSelf(m[0][0], m[1][1]);
        minor[2][2].addEbeMultiplyToSelf(-1, m[0][1], m[0][1], 1);
    }

    /**
     * Sets the minors of the first row of elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setRow0Minors(Vector[][] minor, Vector[][] m, Vector minorStorage) {
        minor[0] = minorStorage.parition(getSubWidth());

        minor[0][1].mapEbeMultiplyToSelf(m[1][1], m[2][2]);
        minor[0][1].addEbeMultiplyToSelf(-1, m[1][2], m[1][2], 1);

        minor[0][1].mapEbeMultiplyToSelf(m[0][1], m[2][2]);
        minor[0][1].addEbeMultiplyToSelf(-1, m[0][2], m[1][2], 1);

        minor[0][2].mapEbeMultiplyToSelf(m[0][1], m[1][2]);
        minor[0][2].addEbeMultiplyToSelf(-1, m[1][1], m[0][2], 1);
    }

    /**
     * Computes the real roots of a cubic equation in the form: x^3 + b x^2 + c
     * x + d = 0
     *
     * TODO: Since this method calls multiple kernels, it would probably be
     * faster if written as a single kernel.
     *
     * @param b Coefficients of the x^2 terms.
     * @param c Coefficients of the x terms.
     * @param d Constant terms.
     * @param roots An array of Vectors where the roots will be stored.
     */
    private static void cubicRoots(Vector b, Vector c, Vector d, Vector roots) {
        KernelManager cos = KernelManager.get("cos"),
                acos = KernelManager.get("acos"),
                sqrt = KernelManager.get("sqrt");

        Vector[] root = roots.parition(3);

        Vector q = root[0];
        q.mapEbeMultiplyToSelf(b, b);
        q.addEbeMultiplyToSelf(2.0 / 27, q, b, 0);
        q.addEbeMultiplyToSelf(-1.0 / 3, b, c, 1);
        q.addToMe(1, d);

        Vector p = d;
        p.addEbeMultiplyToSelf(1.0 / 9, b, b, 0);
        p.addToMe(-1.0 / 3, c); //This is actually p/-3 from wikipedia.

        //c is free for now.  
        Vector theta = c;
        Vector pInverse = p.mapEBEInverse(root[1]); //c is now taken
        sqrt.map(b.getHandle(), pInverse, theta);
        theta.addEbeMultiplyToSelf(-0.5, q, theta, 0);//root[0] is now free (all roots).
        theta.mapEbeMultiplyToSelf(theta, pInverse); //c is now free.
        acos.mapToSelf(b.getHandle(), theta);

        for (int k = 0; k < 3; k++) {
            root[k].set(theta);
            root[k].mapAddToSelf(-2 * Math.PI * k);
        }
        roots.mapMultiplyToSelf(1.0 / 3);
        cos.mapToSelf(b.getHandle(), roots);

        sqrt.mapToSelf(b.getHandle(), p);
        for (Vector r : root) {
            r.addEbeMultiplyToSelf(2, p, r, 0);
            r.addToMe(-1.0 / 3, b);
        }
    }

    /**
     * The ith column of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride column(int i) {
        return new VectorsStride(
                handle,
                data.subArray(i * colDist).getAsBatch(
                        batchArray.stride,
                        batchArray.batchSize,
                        getHeight()
                ),
                getColDist()
        );
    }

    /**
     * The ith column of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride row(int i) {
        return new VectorsStride(
                handle,
                data.subArray(i).getAsBatch(
                        batchArray.stride,
                        batchArray.batchSize,
                        colDist * (getSubWidth() - 1)
                ),
                colDist
        );
    }

    /**
     * Partitions these matrices by column.
     *
     * @return This partitioned by columns.
     */
    public VectorsStride[] columnPartition() {
        VectorsStride[] patritioned = new VectorsStride[subWidth];
        Arrays.setAll(patritioned, i -> column(i));
        return patritioned;

    }

    /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param workSpaceArray The eigen vectors.
     * @return The eigenvectors.
     *
     */
    public MatricesStride computeVecs(VectorsStride eValues, DArray workSpaceArray) {

        MatricesStride eVectors = new MatricesStride(getHandle(), getSubHeight(), getBatchSize());

        try (
                IArray info = IArray.empty(batchArray.batchSize);
                IArray pivot = IArray.empty(getWidth())) {

            MatricesStride workSpace = new MatricesStride(handle, workSpaceArray.getAsBatch(batchArray.stride, batchArray.batchSize, batchArray.subArrayLength), getSubHeight());
            eVectors.getRowVector(getHeight() - 1).fill(1);

            for (int i = 0; i < getHeight(); i++) {
                workSpace.batchArray.set(handle, batchArray, 0, 0, batchArray.length);
                workSpace.computeVec(eValues.getElement(i), eVectors.column(i), info, pivot);
            }
        }

        return eVectors;
    }

    /**
     * Computes an eigenvector for this matrix. This matrix will be changed.
     *
     * @param eValue The eigenvalues.
     * @param eVector Where the eigenvector will be placed.
     * @param info The success of the computations.
     */
    private void computeVec(Vector eValue, VectorsStride eVector, IArray info, IArray pivot) {

        for (int i = 0; i < getHeight(); i++) get(i, i).addToMe(-1, eValue);

        getPointers().LUFactor(handle, pivot, info);

        Vector[][] m = parition();

        Vector x = eVector.getElement(0),
                y = eVector.getElement(1);

        switch (eVector.getSubVecDim()) {//TODO: reduce code redundancy.
            case 3 -> {
                y.set(m[1][2]);
                m[1][1].mapEBEDivide(y);
                x.mapEbeMultiplyToSelf(y, m[0][1]);
                y.mapMultiplyToSelf(-1);
                x.addToMe(-1, m[0][2]);
                m[0][0].mapEBEDivide(x);
            }
            case 2 -> {
                x.set(m[0][1]);
                m[0][0].mapEBEDivide(x);
                x.mapMultiplyToSelf(-1);
            }
            default ->
                throw new UnsupportedOperationException(
                        "ComputeVec only works for 2x2 and 3x3 matrices.  This is a " + eVector.getSubVecDim() + " dimensional eigen vector."
                );
        }
        KernelManager.get("unPivot").map(handle, pivot, getHeight(), eVector.dArray(), eVector.getSubVecDim(), batchArray.batchCount());

    }

    /**
     * Returns this matrix as a set of pointers.
     *
     * @return
     */
    public MatricesPntrs getPointers() {
        if (pointers == null) {
            pointers = new MatricesPntrs(
                    getSubHeight(), getSubWidth(), colDist, batchArray.getPointerArray(handle)
            );
        }
        return pointers;
    }

    private MatricesPntrs pointers;

    /**
     * Returns this matrix as a set of pointers.
     *
     * @param putPointersHere An array where the pointers will be stored.
     * @return
     */
    public MatricesPntrs getPointers(DPointerArray putPointersHere) {

        return new MatricesPntrs(getSubHeight(), getSubWidth(), colDist, putPointersHere.fill(handle, batchArray));

    }

    @Override
    public void close() {
        if (pointers != null) pointers.close();
        if (negOnes3 != null) negOnes3.close();

        super.close();

    }

    /**
     * Gets the matrix at the given index.
     *
     * @param i The index of the desired matrix.
     * @return The matrix at the requested index.
     */
    public Matrix getSubMatrix(int i) {
        return getSubMatrix(0, getHeight(), i * getSubWidth(),
                (i + 1) * getSubWidth());
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < batchArray.batchCount(); i++) {
            sb.append(getSubMatrix(i)).append("\n");
        }
        return sb.toString();
    }

    @Override
    public MatricesStride copy() {
        MatricesStride copy = new MatricesStride(handle, getSubHeight(),
                subWidth, batchArray.stride, batchArray.batchCount());
        return copy(copy);
    }

    /**
     * Copies from this matrix into the proffered matrix.
     *
     * @param copyTo becomes a copy of this matrix.
     * @return the copy.
     */
    public MatricesStride copy(MatricesStride copyTo) {
        if (colDist == getHeight()) {
            copyTo.dArray().set(handle, dArray(), 0, 0, getHeight() * getWidth());
        } else {
            copyTo.addAndSet(1, this, 0, this);
        }
        return copyTo;
    }

    public static void main(String[] args) {
        try (
                Handle handle = new Handle();
                MatricesStride mbs = new MatricesStride(handle, 3, 1);) {

            mbs.dArray().set(handle, new double[]{9, 6, 3, 6, 5, 2, 3, 2, 2});
            Eigen eigen = new Eigen(mbs, true);

            System.out.println("vals = \n" + eigen.values);
            System.out.println("vecs = \n" + eigen.vectors);

        }
    }

    /**
     * The underlying batch array.
     *
     * @return The underlying batch arrays.
     */
    public DStrideArray getBatchArray() {
        return batchArray;
    }

    /**
     * The number of matrices in the batch.
     *
     * @return The number of matrices in the batch.
     */
    public int getBatchSize() {
        return batchArray.batchSize;
    }

    /**
     * Returns a matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     *
     * @param startRow The row the submatrices start on.
     * @param endRowExclsve The row the submatrices end on, exclusive.
     * @return A matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     */
    public MatricesStride subMatrixRows(int startRow, int endRowExclsve) {
        DStrideArray subMData = data.subArray(startRow, (getWidth() - 1) * colDist).getAsBatch(batchArray.stride, getBatchSize(), batchArray.subArrayLength);
        return new MatricesStride(handle, subMData, endRowExclsve - startRow, subWidth, colDist);
    }
    
    /**
     * Returns a matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     *
     * @param startCol The col the submatrices start on.
     * @param endColExclsve The row the submatrices end on, exclusive.
     * @return A matrices stride where each matrix is a sub matrix of one of the
     * matrices in this.
     */
    public MatricesStride subMatrixCols(int startCol, int endColExclsve) {
        DStrideArray subMData = data.subArray(startCol*colDist)
                .getAsBatch(batchArray.stride, getBatchSize(), batchArray.subArrayLength);
        
        return new MatricesStride(handle, subMData, getSubHeight(), endColExclsve - startCol, colDist);
    }
    
    
    /**
     * A sub batch of this batch.
     * @param start The index of the first subMatrix.
     * @param length One after the index of the last submatrix.
     * @return A subbatch.
     */
    public MatricesStride subBatch(int start, int length){
        return new MatricesStride(
                handle, 
                batchArray.subBatch(start, length), 
                getSubHeight(), 
                getSubWidth(), 
                colDist
        );
        
    }

    @Override
    public MatricesStride fill(double scalar) {
        super.fill(scalar); 
        return this;
    }
    
    
}
