package JCudaWrapper.algebra;

import JCudaWrapper.algebra.Eigen;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DPointerArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import java.util.Arrays;
import org.apache.commons.math3.exception.DimensionMismatchException;
import JCudaWrapper.resourceManagement.Handle;
import java.awt.Dimension;

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
public class MatricesStride extends TensorOrd3Stride implements ColumnMajor, AutoCloseable {

    /**
     * Constructor for creating a batch of strided matrices. Each matrix is
     * stored with a specified stride and the batch is represented as a single
     * contiguous block in memory.
     *
     * @param handle The handle for resource management and creating this
     * matrix. It will stay with this matrix instance.
     * @param height The number of rows (height) in each submatrix.
     * @param width The number of columns (width) in each submatrix.
     * @param stride The number of elements between the first elements of
     * consecutive submatrices in the batch.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int height, int width, int stride, int batchSize) {
        this(
                handle,
                DArray.empty(DStrideArray.totalDataLength(stride, width * height, batchSize)),
                height,
                width,
                height,
                stride,
                batchSize
        );
    }

    /**
     * Creates a simple batch matrix with coldDist = height.
     *
     * @param handle
     * @param data The length of this data should be width*height.
     * @param height The height of this matrix and the sub matrices.
     * @param width The width of each sub matrix.
     * @param colDist The distance between the first element of each column.
     * @param strideSize How far between the first elements of each matrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, DArray data, int height, int width, int colDist, int strideSize, int batchSize) {
        super(handle, height, width, 1, colDist, height*width, strideSize, batchSize, data);
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
     * @param height The number of rows (height) in each submatrix.
     * @param width The number of columns (width) in each submatrix.
     * @param batchSize The number of matrices in this batch.
     */
    public MatricesStride(Handle handle, int height, int width, int batchSize) {
        this(handle, height, width, height * width, batchSize);
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
    public Vector matIndices(int i, int j) {
        return new Vector(handle, data.subArray(index(i, j), (getBatchSize() - 1) * getStrideSize() + 1), data.stride);
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
        Vector[][] all = new Vector[height][width];
        for (int i = 0; i < height; i++) {
            int row = i;
            Arrays.setAll(all[row], col -> matIndices(row, col));
        }
        return all;
    }

    /**
     * Performs matrix multiplication on the batches of matrices, and adds them
     * to this matrix. This method multiplies matrix batches {@code a} and
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
     * @param timesThis The scaling factor applied to the result matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatricesStride addProduct(boolean transposeA, boolean transposeB, double timesAB, MatricesStride a, MatricesStride b, double timesThis) {

        Dimension aDim = new Dimension(transposeA ? a.height : a.width, transposeA ? a.width : a.height);
        Dimension bDim = new Dimension(transposeB ? b.height : b.width, transposeB ? b.width : b.height);

        if (aDim.width != bDim.height || height != aDim.height || width != bDim.width)
            throw new DimensionMismatchException(aDim.width, bDim.height);

        data.addProduct(handle, transposeA, transposeB,
                aDim.height, aDim.width, bDim.width,
                timesAB,
                a.data, a.colDist,
                b.data, b.colDist,
                timesThis, colDist
        );

        return this;
    }

    /**
     * Multiplies each matrix in the batch by it's coresponding scalar in the
     * vector.
     *
     * @param scalars the ith element is multiplied by the ith matrix.
     * @return this
     * @throws DimensionMismatchException if the dimensions of matrices
     * {@code a} and {@code b} are incompatible for multiplication.
     */
    public MatricesStride multiply(Vector scalars) {

        KernelManager.get("prodScalarMatrixBatch").vectorBatchMatrix(handle, scalars, this);

        return this;
    }

    /**
     * Multiplies all the matrices in this set by the given scalar.
     *
     * @param scalar multiply all these matrices by this scalar.
     * @param workSpace Should be width * height.
     * @return this
     */
    public MatricesStride multiply(double scalar, DArray workSpace) {
        add(false, scalar, this, 0, workSpace);
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

        if (height != 2)
            throw new IllegalArgumentException("compute vals 2x2 can only be called on a 2x2 matrix.  These matrices are " + height + "x" + width);

        VectorsStride eVals = new VectorsStride(handle, 2, getBatchSize(), 2, 1);
        Vector[] eVal = eVals.vecPartition();//val[0] is the first eigenvalue, val[1] the seond, etc...

        Vector[][] m = parition();

        Vector trace = workSpace.getSubVector(0, data.batchCount());

        trace.set(m[1][1]).add(1, m[0][0]);//= a + d

        eVal[1].ebeSetProduct(m[0][1], m[0][1]).ebeAddProduct(m[0][0], m[1][1], -1);// = ad - c^2

        eVal[0].ebeSetProduct(trace, trace).add(-4, eVal[1]);//=(d + a)^2 - 4(ad - c^2)

        KernelManager.get("sqrt").mapToSelf(handle, eVal[0]);//sqrt((d + a)^2 - 4(ad - c^2))

        eVal[1].set(trace);
        eVal[1].add(-1, eVal[0]);
        eVal[0].add(1, trace);

        eVals.data.multiply(handle, 0.5, 1);

        return eVals;
    }

    /**
     * Computes the eigenvalues for a set of symmetric 3x3 matrices. If this
     * batch is not such a set then this method should not be called.
     *
     * @param workSpace Should have length equal to 3*batchSize.
     * @return The eigenvalues.
     *
     */
    //m := a, d, g, d, e, h, g, h, i = m00, m10, m20, m01, m11, m21, m02, m12, m22
    //p := tr m = a + e + i
    //q := (p^2 - norm(m)^2)/2 where norm = a^2 + d^2 + g^2 + d^2 + ...
    // solve: lambda^3 - p lambda^2 + q lambda - det m        
    public VectorsStride computeVals3x3(double tolerance) {

        if (height != 3)
            throw new IllegalArgumentException("computeVals3x3 can only be called on a 3x3 matrix.  These matrices are " + height + "x" + width);

        VectorsStride vals = new VectorsStride(handle, height, getBatchSize(), height, 1);
        
        KernelManager.get("eigenValsBatch").map(handle, batchSize, dArray(), vals.dArray().pToP(), DArray.cpuPoint(tolerance));
        
        return vals;
//        Vector[] work = workSpace.parition(3);
//
//        Vector[][] m = parition();
//
//        Vector[][] minor = new Vector[3][3];
//
//        Vector negTrace = negativeTrace(work[0]);//work[0] is taken
//
//        setDiagonalMinors(minor, m, vals); //vals are now all taken.
//        Vector C = work[1].fill(0);
//        for (int i = 0; i < 3; i++) C.add(1, minor[i][i]);
//
//        setRow0Minors(minor, m, vals);
//        Vector det = work[2];
//        det.ebeSetProduct(m[0][0], minor[0][0]);
//        det.addEbeProduct(-1, m[0][1], minor[0][1], 1);
//        det.addEbeProduct(-1, m[0][2], minor[0][2], -1);
//
//        cubicRoots(negTrace, C, det, vals);
//
//        return vals;
    }

    private static DArray negOnes3;

    /**
     * A vector of 3 negative ones.
     *
     * @return A vector of 3 negative ones.
     */
    private static VectorsStride negOnes3(Handle handle, int batchSize) {
        if (negOnes3 == null) negOnes3 = DArray.empty(3).fill(handle, -1, 1);
        return new VectorsStride(handle, negOnes3, 1, 3, 0, batchSize);
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

        VectorsStride diagnols = new VectorsStride(handle, data, 4, 3, data.stride, data.batchSize);

        return traceStorage.setBatchVecVecMult(
                diagnols,
                negOnes3(handle, data.batchSize)
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

        Vector[] storagePartition = minorStorage.vecPartition();

        for (int i = 0; i < minor.length; i++)
            minor[i][i] = storagePartition[i];

        minor[0][0].ebeSetProduct(m[1][1], m[2][2]);
        minor[0][0].addEbeProduct(-1, m[1][2], m[1][2], 1);

        minor[1][1].ebeSetProduct(m[0][0], m[2][2]);
        minor[1][1].addEbeProduct(-1, m[0][2], m[0][2], 1);

        minor[2][2].ebeSetProduct(m[0][0], m[1][1]);
        minor[2][2].addEbeProduct(-1, m[0][1], m[0][1], 1);
    }

    /**
     * Sets the minors of the first row of elements.
     *
     * @param minor Where the new minors are to be stored.
     * @param m The elements of the matrix.
     * @param minorStorage A space where the minors can be stored.
     */
    private void setRow0Minors(Vector[][] minor, Vector[][] m, VectorsStride minorStorage) {
        minor[0] = minorStorage.vecPartition();

        minor[0][1].ebeSetProduct(m[1][1], m[2][2]);
        minor[0][1].addEbeProduct(-1, m[1][2], m[1][2], 1);

        minor[0][1].ebeSetProduct(m[0][1], m[2][2]);
        minor[0][1].addEbeProduct(-1, m[0][2], m[1][2], 1);

        minor[0][2].ebeSetProduct(m[0][1], m[1][2]);
        minor[0][2].addEbeProduct(-1, m[1][1], m[0][2], 1);
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
    private static void cubicRoots(Vector b, Vector c, Vector d, VectorsStride roots) {
        KernelManager cos = KernelManager.get("cos"),
                acos = KernelManager.get("acos"),
                sqrt = KernelManager.get("sqrt");

        Vector[] root = roots.vecPartition();

        Vector q = root[0];
        q.ebeSetProduct(b, b);
        q.addEbeProduct(2.0 / 27, q, b, 0);
        q.addEbeProduct(-1.0 / 3, b, c, 1);
        q.add(1, d);

        Vector p = d;
        p.addEbeProduct(1.0 / 9, b, b, 0);
        p.add(-1.0 / 3, c); //This is actually p/-3 from wikipedia.

        //c is free for now.  
        Vector theta = c;
        Vector pInverse = root[1].fill(1).ebeDivide(p); //c is now taken               
        sqrt.map(b.getHandle(), pInverse, theta);

        theta.addEbeProduct(-0.5, q, theta, 0);//root[0] is now free (all roots).
        theta.ebeSetProduct(theta, pInverse); //c is now free.
        acos.mapToSelf(b.getHandle(), theta);

        for (int k = 0; k < 3; k++) root[k].set(theta).add(-2 * Math.PI * k);

        roots.data.multiply(b.getHandle(), 1.0 / 3, 1);
        cos.mapToSelf(b.getHandle(), roots.data);

        sqrt.mapToSelf(b.getHandle(), p);
        for (Vector r : root) {
            r.addEbeProduct(2, p, r, 0);
            r.add(-1.0 / 3, b);
        }
    }

    /**
     * The ith column of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride column(int i) {
        return new VectorsStride(handle, data.subArray(i * colDist), 1, height, data.stride, data.batchSize);
    }

    /**
     * The ith row of each submatrix.
     *
     * @param i The index of the desired column.
     * @return The ith column of each submatrix.
     */
    public VectorsStride row(int i) {
        return new VectorsStride(handle, data.subArray(i), colDist, width, data.stride, data.batchSize);
    }

    /**
     * Partitions these matrices by column.
     *
     * @return This partitioned by columns.
     */
    public VectorsStride[] columnPartition() {
        VectorsStride[] patritioned = new VectorsStride[width];
        Arrays.setAll(patritioned, i -> column(i));
        return patritioned;

    }

    /**
     * Adds dimensions like batchsize and width to the given data.
     *
     * @param addDimensions data in need of batch dimensions.
     * @return The given data with this's dimensions.
     */
    public MatricesStride copyDimensions(DArray addDimensions) {
        return new MatricesStride(
                handle,
                addDimensions,
                height,
                width,
                colDist,
                data.stride,
                data.batchSize
        );
    }

    /**
     * Adds dimensions like batchsize and width to the given data.
     *
     * Stride and batch size are taken from add Dimensions, the rest of the
     * dimensions from this.
     *
     * @param addDimensions data in need of batch dimensions.
     * @return The given data with this's dimensions.
     */
    public MatricesStride copyDimensions(DStrideArray addDimensions) {
        return new MatricesStride(
                handle,
                addDimensions,
                height,
                width,
                colDist,
                addDimensions.stride,
                addDimensions.batchSize
        );
    }

    /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param workSpaceDArray Should have width * as many elements as there are in this.
     * @param workSpaceIArray Should be of length batch size.
     * @param tolerance What is considered 0
     * @return The eigenvectors.
     *
     */
    public MatricesStride computeVecs(VectorsStride eValues, DArray workSpaceDArray, IArray workSpaceIArray, double tolerance) {

        MatricesStride eVectors = copyDimensions(DArray.empty(data.length));
        
        KernelManager.get("eigenVecBatch").map(handle, 
                eValues.dim()*eValues.batchSize,                
                data,
                IArray.cpuPoint(height),
                eVectors.dArray().pToP(),
                IArray.cpuPoint(eVectors.width),
                eValues.dArray().pToP(),
                workSpaceDArray.pToP(),
                workSpaceIArray.pToP(),
                DArray.cpuPoint(tolerance)
        );
        return eVectors;
    }
        /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param workSpaceArray Should have 3* as many elements as there are in this.
     * @param tolerance What is considered 0
     * @return The eigenvectors.
     *
     */        /**
     * Computes the eigenvector for an eigenvalue. The matrices must be
     * symmetric positive definite. TODO: If any extra memory is available, pass
     * it here!
     *
     * @param eValues The eigenvalues, organized by sets per matrix.
     * @param workSpaceArray Should have 3* as many elements as there are in this.
     * @param tolerance What is considered 0
     * @return The eigenvectors.
     *
     */
    public MatricesStride computeVecs3x3(VectorsStride eValues, DArray workSpaceArray, double tolerance) {

        MatricesStride eVectors = copyDimensions(DArray.empty(data.length));
        
        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVecs3x3() " + batchSize*3);
        
        KernelManager.get("eigenVecBatch3x3").map(handle,
                batchSize*3,
                data,
                IArray.cpuPoint(colDist),
                eVectors.dArray().pToP(),
                IArray.cpuPoint(eVectors.colDist),
                eValues.dArray().pToP(),
                workSpaceArray.pToP(),
                DArray.cpuPoint(tolerance)
        );

        return eVectors;
    }

    /**
     * Computes an eigenvector for this matrix. This matrix will be changed.
     *
     * @param eValue The eigenvalues.
     * @param eVector Where the eigenvector will be placed.
     * @param tolerance what is considered 0.
     */
    private void computeVec(Vector eValue, VectorsStride eVector, double tolerance) {

        for (int i = 0; i < height; i++) matIndices(i, i).add(-1, eValue);
        
        KernelManager.get("nullSpace1dBatch").map(handle, 
                getBatchSize(),
                data, colDist,
                eVector.data, eVector.getStrideSize(),
                IArray.cpuPoint(width), 
                DArray.cpuPoint(tolerance)
        );

    }

    /**
     * Returns this matrix as a set of pointers.
     *
     * @return
     */
    public MatricesPntrs getPointers() {
        if (pointers == null) {
            pointers = new MatricesPntrs(
                    height, width, colDist, data.getPointerArray(handle)
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

        return new MatricesPntrs(height, width, colDist, putPointersHere.fill(handle, data));

    }

    @Override
    public void close() {
        data.close();
    }

    /**
     * Gdets the matrix at the given index.
     *
     * @param i The index of the desired matrix.
     * @return The matrix at the requested index.
     */
    public Matrix getMatrix(int i) {
        return new Matrix(handle, data.getBatchArray(i), height, width, colDist);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < data.batchCount(); i++)
            sb.append(getMatrix(i).toString()).append("\n\n");

        return sb.toString();
    }

    /**
     * A copy of this matrices stride.
     *
     * @return A copy of this matrices stride.
     */
    public MatricesStride copy() {
        return copyDimensions(data.copy(handle));
    }

    /**
     * Copies from this matrix into the proffered matrix. Note, underlying data
     * is copied even if it does not appear in this matrix. TODO: fix this so
     * that unused underlying data is not copied.
     *
     * @param copyTo becomes a copy of this matrix.
     * @return the copy.
     */
    public MatricesStride copy(DArray copyTo) {
        copyTo.set(handle, data, 0, 0, data.length);

        return copyDimensions(copyTo);
    }

    public static void main(String[] args) {
        try (
                Handle handle = new Handle();
                DArray array = new DArray(handle, 0, 2, 0, 3/*, 9, 11, 11, 12, 5, 6, 6, 8, 1, 0, 0, 0*/)) {

            int numMatrices = 1;

            MatricesStride ms = new MatricesStride(handle, array, 2, 2, 2, 4, numMatrices);

            Matrix[] m = new Matrix[1];
            Arrays.setAll(m, i -> new Matrix(handle, array.subArray(i * 4), 2, 2));
            for (int i = 0; i < m.length; i++) m[i].power(2);

            System.out.println("matrices:\n" + ms);

            try (Eigen eigen = new Eigen(ms, 1e-13)) {

                System.out.println("Eigen values:\n" + eigen.values);

                System.out.println("Eigen vectors:\n" + eigen.vectors);

                System.out.println("verifying:\n");
                for (int matInd = 0; matInd < ms.getBatchSize(); matInd++)
                    for (int j = 0; j < ms.height; j++)
                        System.out.println(ms.getMatrix(matInd) + " * " + eigen.vectors.getMatrix(matInd).getColumn(j) + " * " + "1/" + eigen.values.getVector(matInd).get(j) + " = "
                                + eigen.vectors.getMatrix(matInd).getColumn(j).addProduct(
                                        false,
                                        1 / eigen.values.getVector(matInd).get(j),
                                        ms.getMatrix(matInd),
                                        eigen.vectors.getMatrix(matInd).getColumn(j),
                                        0
                                ));
            }
        }
    }

//    public static void main(String[] args) {
//        try(Handle hand = new Handle(); DArray da = new DArray(hand, 1,0,0,0)){
//            Matrix m = new Matrix(hand, da, 2, 2);
//            
//            System.out.println("matrix = \n" + m + "\n");
//            
//            Eigen eigen = new Eigen(m.repeating(1));
//            
//            System.out.println("values: " + eigen.values + "\n");
//            System.out.println("vectors:\n" + eigen.vectors);
//            
//        }
//    }
    /**
     * The underlying batch array.
     *
     * @return The underlying batch arrays.
     */
    public DStrideArray getBatchArray() {
        return data;
    }

    /**
     * The number of matrices in the batch.
     *
     * @return The number of matrices in the batch.
     */
    public int getBatchSize() {
        return data.batchSize;
    }

    /**
     * The stride size.
     *
     * @return The stride size.
     */
    public int getStrideSize() {
        return data.stride;
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
        return new MatricesStride(handle, data, 1, width, colDist, 1, height);
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
        return new MatricesStride(handle, data, height, 1, colDist, colDist, width);
    }

    /**
     * A sub batch of this batch.
     *
     * @param start The index of the first subMatrix.
     * @param length One after the index of the last submatrix.
     * @return A subbatch.
     */
    public MatricesStride subBatch(int start, int length) {
        return copyDimensions(data.subBatch(start, length));

    }

    /**
     * If this matrices stride were to be represented as a single matrix, then
     * this would be its width.
     *
     * @return The width of a suggested representing all inclusive matrix.
     */
    private int totalWidth() {
        return data.stride * data.batchSize + width;
    }

    /**
     * Fills all elements of these matrices with the given values. This method
     * alocated two gpu arrays, so it probably should not be called for many
     * batches.
     *
     * @param scalar To fill these matrices.
     * @return This.
     */
    public MatricesStride fill(double scalar) {
        if (data.stride <= colDist * width) {
            if (colDist == height) data.fill(handle, scalar, 1);
            else if (height == 1) data.fill(handle, scalar, colDist);
            else data.fillMatrix(handle, height, totalWidth(), colDist, scalar);
        } else {
            try (DArray workSpace = DArray.empty(width * width)) {

                MatricesStride empty = new Matrix(handle, workSpace, height, width).repeating(data.batchSize);

                add(false, scalar, empty, 0, workSpace);
            }
        }
        return this;
    }


    /**
     * The handle used for this's operations.
     *
     * @return The handle used for this's operations.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Adds matrices to these matrices.
     *
     * @param transpose Should toAdd be transposed.
     * @param timesToAdd Multiply toAdd before adding.
     * @param toAdd The matrices to add to these.
     * @param timesThis multiply this before adding.
     * @param workSpace workspace should be width^2 length.
     * @return
     */
    public MatricesStride add(boolean transpose, double timesToAdd, MatricesStride toAdd, double timesThis, DArray workSpace) {

        Matrix id = Matrix.identity(handle, width, workSpace);

        addProduct(transpose, false, timesToAdd, toAdd, id.repeating(data.batchSize), timesThis);
        return this;
    }

    /**
     * Row operation that convert this matrix into a diagonal matrix.
     *
     * @return These matrices.
     */
    public MatricesStride diagnolize(DArray pivot) {
        throw new UnsupportedOperationException("This method is not yet written.");
    }

    @Override
    public int getColDist() {
        return colDist;
    }

}

////Former eVec method.
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() After eigen subtraction:\n" + toString());
//        getPointers().LUFactor(handle, pivot, info);
//
////        try(DArray ld = DArray.empty(4); DArray ud = DArray.empty(4)){
////            //delte this hole section.  Its just for debugging.
////            Matrix l = getSubMatrix(0).lowerLeftUnitDiagonal(ld);
////            Matrix u = getSubMatrix(0).upperRight(ud);
////            System.out.println("checking LU product: \n" + l.multiplyAndSet(l, u));
////        }
////        
////        
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() pivot\n" + pivot.toString());
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec() after LU:\n" + toString());
//
//        JCudaWrapper.algebra.Vector[][] m = parition();
//
////        System.out.println("JCudaWrapper.algebra.MatricesStride.computeVec()  m = " + Arrays.deepToString(m));
//        eVector.get(height - 2)
//                .set(m[width - 2][height - 1])
//                .ebeDivide(m[width - 2][height - 2])
//                .multiply(-1);
//
//        if (eVector.getSubVecDim() == 3)
//            eVector.getElement(0)
//                    .set(eVector.getElement(1))
//                    .multiply(-1, m[0][1])
//                    .add(-1, m[0][2])
//                    .ebeDivide(m[0][0]);
//
////        System.out.println("Before pivoting: " + eVector);
////        KernelManager.get("unPivotVec").map(//TODO: understand why unpivoting seems to give the wrong answer, and not unpivoting seems to get it right.
////                        handle, 
////                        pivot, 
////                        height, 
////                        eVector.dArray(), 
////                        eVector.inc(), 
////                        data.batchCount(),
////                        IArray.cpuPointer(data.stride)
////                );
////        System.out.println("After pivoting:  " + eVector);
