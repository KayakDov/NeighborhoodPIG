package JCudaWrapper.algebra;

import java.util.Arrays;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.exception.NotPositiveException;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DSingleton;

/**
 * The {@code Vector} class extends {@code RealVector} and represents a vector
 * stored on the GPU. It relies on the {@code DArray} class for data storage and
 * the {@code Handle} class for JCublas operations.
 *
 * Vectors are horizontal matrices.
 */
public class Vector extends Matrix {

    /**
     * Constructs a new {@code Vector} from an existing data pointer on the GPU.
     * Do not use this constructor for vectors with inc == 0.
     *
     * @param data The {@code DArray} storing the vector on the GPU.
     * @param inc The increment between elements of the data that make of this
     * vector. If 0 is passed, methods that usde JCUDA matrix operations will
     * not work, while methods that use JCuda vector operations will. TODO: make
     * this better.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, DArray data, int inc) {
        super(handle, data, 1, Math.ceilDiv(data.length, inc), inc);
    }

    /**
     * Constructs a new {@code Vector} from a 1D array.
     *
     * @param array The array storing the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, double... array) {
        this(handle, new DArray(handle, array), 1);
    }

    /**
     * Constructs a new empty {@code Vector} of specified length.
     *
     * @param length The length of the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, int length) {
        this(handle, DArray.empty(length), 1);
    }

    /**
     * Gets the element at the given index.
     *
     * @param index The index of the desired element.
     * @return The element at the given index.
     * @throws OutOfRangeException If the element is out of range.
     */
    public double get(int index) throws OutOfRangeException {
        return data.get(index * inc()).getVal(handle);
    }

    /**
     * Sets the element at the given index.
     *
     * @param index The index whose element is to be set.
     * @param value The value to be placed at index.
     * @throws OutOfRangeException
     */
    public void set(int index, double value) throws OutOfRangeException {
        data.set(handle, index * inc(), value);
    }

    /**
     * The dimension of the vector. The number of elements in it.
     *
     * @return The dimension of the vector. The number of elements in it.
     */
    public int dim() {
        return inc() == 0 ? 1 : Math.ceilDiv(data.length, inc());
    }

    /**
     * Adds another vector times a scalar to this vector, changing this vector.
     *
     * @param mult A scalar to be multiplied by @code{v} before adding it to
     * this vector.
     * @param v The vector to be added to this vector.
     * @return This vector.
     */
    public Vector addToMe(double mult, Matrix v) {
        data.addToMe(handle, mult, v.data, v.colDist, inc());
        return this;
    }

    /**
     * Adds the scalar to every element in this vector.
     *
     * @param scalar To be added to every element in this vector.
     * @return this.
     */
    @Override
    public Vector addToMe(double scalar) {
        data.addToMe(handle, scalar, DSingleton.oneOne, 0, 1);
        return this;

    }

    /**
     * multiplies this array by the scalar.
     *
     * @param scalar to multiply this array.
     * @return this.
     */
    public Vector multiplyMe(double scalar) {
        data.multMe(handle, scalar, inc());
        return this;
    }

    /**
     * Sets all the values in this vector to that of the scalar.
     *
     * @param scalar The new value to fill this vector.
     * @return This vector.
     */
    public Vector fill(double scalar) {
        if (scalar == 0 && inc() == 1) {
            data.fill0(handle);
        } else {
            data.fill(handle, scalar, inc());
        }
        return this;
    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @see Vector#dotProduct(org.apache.commons.math3.linear.RealVector)
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public double dotProduct(Vector v) {

        return data.dot(handle, v.data, v.inc(), inc());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector copy() {
        if (inc() == 1) {
            return new Vector(handle, data.copy(handle), inc());
        }
        Vector copy = new Vector(handle, dim());
        copy.data.set(handle, data, 0, 0, 1, inc(), dim());
        return copy;
    }

    /**
     * Computes the element-wise product of this vector and another vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeMultiplyAndSet(Vector a, Vector b) {

        return addEbeMultiplyToSelf(a, b, 0);

    }

    /**
     * multiplies this vector by the given scalar and vector (element by
     * element).
     *
     * @param scalar The scalar times this.
     * @param a The vector that will be ebe times this.
     * @return this.
     */
    public Vector multiplyMe(double scalar, Vector a) {
        return addEbeMultiplyToSelf(scalar, a, this, 0);
    }

    /**
     * Computes the element-wise product of this vector and another vector, and
     * adds it to this vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param timesThis Multiply this matrix before adding the product of a and
     * b to it.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector addEbeMultiplyToSelf(Vector a, Vector b, double timesThis) {
        return Vector.this.addEbeMultiplyToSelf(1, a, b, timesThis);
    }

    /**
     * Computes the element-wise product of this vector and another vector, and
     * adds it to this vector.
     *
     * @param a The first vector.
     * @param b The second vector.
     * @param timesAB A scalar to multiply by a and b.
     * @param timesThis multiply this vector before adding the product of a and
     * b.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector addEbeMultiplyToSelf(double timesAB, Vector a, Vector b, double timesThis) {

        data.multSymBandMatVec(handle, true,
                dim(), 0,
                timesAB, a.data, a.inc(),
                b.data, b.inc(),
                timesThis, inc()
        );

        return this;
    }

    /**
     * Computes the element-wise product of this vector and another vector, and
     * adds it to this vector.
     *
     * @param workSpace A space to work in. It should be at least the size of
     * this vector.
     * @param a The first vector.
     * @param timesAB A scalar to multiply by a and b.
     * @param timesThis multiply this vector before adding the product of a and
     * b.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector addEbeMultiplyToSelf(Vector workSpace, double timesAB, double timesThis, Vector... a) {

        multiplyMe(timesThis);

        if (a.length == 0) {
            return this;
        }
        if (a.length == 1) {
            return Vector.this.addEbeMultiplyToSelf(timesAB, this, a[0], 1);
        }
        if (a.length == 2) {
            return Vector.this.addEbeMultiplyToSelf(timesAB, a[0], a[1], 1);
        }

        workSpace.addEbeMultiplyToSelf(timesAB, a[0], a[1], 0);

        for (int i = 2; i < a.length; i++) {
            workSpace.addEbeMultiplyToSelf(workSpace, a[i], 0);
        }

        return addToMe(1, workSpace);

    }

    /**
     * Element by element division. Like most methods, it changes this vector.
     *
     * @param denominator the denominator.
     * @return this
     */
    public Vector ebeDivide(Vector denominator) {

        try {
            data.solveTriangularBandedSystem(handle, true, false, false,
                    denominator.dim(), 0, denominator.data, denominator.inc(), inc());
        } catch (Exception e) {
            if (Arrays.stream(denominator.toArray()).anyMatch(i -> i == 0))
                throw new ArithmeticException("Division by 0.");
            else throw e;

        }

        return this;
    }

    /**
     * A sub vector of this one.
     *
     * @param begin The index of this vector to begin the subvector.
     * @param length The number of elements in the subvector.
     * @return A subvector of this vector.
     * @throws NotPositiveException
     * @throws OutOfRangeException
     */
    public Vector getSubVector(int begin, int length) throws NotPositiveException, OutOfRangeException {
        return getSubVector(begin, length, 1);
    }

    /**
     * Returns a sub vector of this one. The vector is a shallow copy/ copy by
     * reference, changes made to the new vector will affect this one and vice
     * versa.
     *
     * @param begin Where the vector begins.
     * @param length The length of the new vector.The number of elements in the
     * vector.
     * @param increment The stride step of the new vector. For example, if this
     * value is set to 2, then the new vector will contain every other element
     * of this vector.
     * @return A sub vector of this vector.
     * @throws NotPositiveException
     * @throws OutOfRangeException
     */
    public Vector getSubVector(int begin, int length, int increment) throws NotPositiveException, OutOfRangeException {
        return new Vector(
                handle,
                data.subArray(begin * inc(), inc() * increment * (length - 1) + 1),
                inc() * increment
        );
    }

    /**
     *
     * @see Vector#setSubVector(int, org.apache.commons.math3.linear.RealVector)
     * Sets a subvector of this vector starting at the specified index using the
     * elements of the provided {@link Vector}.
     *
     * This method modifies this vector by copying the elements of the given
     * {@link Vector} into this vector starting at the position defined by the
     * index, with the length of the subvector determined by the dimension of
     * the given vector.
     *
     * @param i The starting index where the subvector will be set, scaled by
     * the increment (inc).
     * @param rv The {@link Vector} whose elements will be copied into this
     * vector.
     * @throws OutOfRangeException If the index is out of range.
     */
    public void setSubVector(int i, Vector rv) throws OutOfRangeException {
        data.set(handle, rv.data, i * inc(), 0, inc(), rv.inc(), rv.dim());
    }

    /**
     * Sets a portion of this vector to the contents of the given matrix.
     * Specifically, the method inserts the columns of the matrix into this
     * vector starting at the specified index and offsets each column by the
     * height of the matrix.
     *
     * @param toIndex the starting index in this vector where the subvector
     * (matrix columns) will be inserted
     * @param m the matrix whose columns are used to set the subvector in this
     * vector
     *
     * @throws IndexOutOfBoundsException if the specified index or resulting
     * subvector extends beyond the vector's bounds
     *
     *///TODO: This method can be made faster with multi threading (multiple handles)
    public void setSubVector(int toIndex, Matrix m) {
        for (int mCol = 0; mCol < dim(); mCol++) {
            setSubVector(toIndex + mCol * m.getHeight(), m.getColumn(mCol));
        }
    }

    /**
     * Sets a portion of this vector to the contents of the given Vector.
     *
     *
     * @param v The vector to copy from.
     * @return this
     */
    public Vector set(Vector v) {
        setSubVector(0, v);
        return this;
    }

    /**
     * Computes the cosine of the angle between this vector and the argument.
     *
     * @see Vector#cosine(org.apache.commons.math3.linear.RealVector)
     * @param other Vector
     * @return the cosine of the angle between this vector and v.
     */
    public double cosine(Vector other) {

        return dotProduct(other) / norm() * other.norm();
    }

    /**
     * The distance between this vector and another vector.
     *
     * @param v The other vector.
     * @param workSpace Should be as long as these vectors.
     * @return The distance to v.
     * @throws DimensionMismatchException
     */
    public double getDistance(Vector v, Vector workSpace) throws DimensionMismatchException {
        workSpace.fill(0);
        workSpace.addAndSet(1, v, -1, this);
        return workSpace.norm();
    }

    /**
     * The L_1 norm.
     */
    public double getL1Norm() {
        return data.sumAbs(handle, dim(), inc());
    }

    /**
     * The L_infinity norm
     */
    public double getLInfNorm() {
        return get(data.argMaxAbs(handle, dim(), inc()));
    }

    /**
     * Finds the index of the minimum or maximum element of the vector. This
     * method creates its own workspace equal in size to this.
     *
     * @param isMax True to find the argMaximum, false for the argMin.
     * @return The argMin or argMax.
     */
    private int getMinMaxInd(boolean isMax) {
        int argMaxAbsVal = data.argMaxAbs(handle, dim(), inc());
        double maxAbsVal = get(argMaxAbsVal);
        if (maxAbsVal == 0) {
            return 0;
        }
        if (maxAbsVal > 0 && isMax) {
            return argMaxAbsVal;
        }
        if (maxAbsVal < 0 && !isMax) {
            return argMaxAbsVal;
        }

        try (Vector sameSign = copy().addToMe(maxAbsVal)) {
            return sameSign.data.argMinAbs(handle, dim(), inc());
        }
    }

    /**
     * The index of the minimum element.
     */
    public int minIndex() {
        return getMinMaxInd(false);
    }

    /**
     * The minimum value.
     */
    public double getMinValue() {
        return get(minIndex());
    }

    /**
     * The maximum index.
     */
    public int maxIndex() {
        return getMinMaxInd(true);
    }

    /**
     * The maximum value.
     */
    public double maxValue() {
        return get(maxIndex());
    }

    /**
     * @param v The vector with which this one is creating an outer product.
     * @param placeOuterProduct should have at least v.dim() * dim() elements.
     * @return The outer product. A new matrix.
     * @see Vector#outerProduct(org.apache.commons.math3.linear.RealVector)
     */
    public Matrix outerProduct(Vector v, DArray placeOuterProduct) {
        placeOuterProduct.outerProd(handle, dim(), v.dim(), 1, data, inc(), v.data, v.inc(), dim());
        return new Matrix(handle, placeOuterProduct, dim(), v.dim()).fill(0);
    }

    /**
     * @see Vector#projection(org.apache.commons.math3.linear.RealVector)
     *
     * @param v project onto.
     * @return The projection.
     *
     */
    public Vector projection(Vector v) throws DimensionMismatchException, MathArithmeticException {
        double[] dots = new double[2];

        data.dot(handle, v.data, v.inc(), inc(), dots, 0);
        v.data.dot(handle, v.data, v.inc(), inc(), dots, 1);

        return v.multiplyMe(dots[0] / dots[1]);
    }

    /**
     * The cpu array that is a copy of this gpu vector.
     *
     * @return the array in the cpu.
     */
    public double[] toArray() {
        double[] to = new double[dim()];
        data.get(handle, to, 0, 0, 1, inc(), dim());
        return to;
    }

    /**
     * Turn this vector into a unit vector.
     */
    public void unitize() throws MathArithmeticException {
        multiplyMe(1 / norm());
    }

    /**
     * The data underlying this vector.
     *
     * @return The underlying data from this vector.
     */
    public DArray dArray() {
        return data;
    }

    /**
     * A matrix representing the data underlying this Vector. Note, depending on
     * inc and colDist, the new matrix may have more or fewere elements than
     * this vector.
     *
     * @param height The height of the new matrix.
     * @param width The width of the new matrix.
     * @param colDist The disance between the first element of each column.
     * @return
     */
    public Matrix asMatrix(int height, int width, int colDist) {
        return new Matrix(handle, data, height, width, colDist);
    }

    /**
     * A matrix representing the data underlying this Vector. Note, depending on
     * inc and colDist, the new matrix may have more or fewere elements than
     * this vector.
     *
     * @param height The height of the new matrix. It should be divisible by the
     * number of elements in the underlying data.
     * @return A matrix containing the elements in the underlying data of this
     * vector.
     */
    public Matrix asMatrix(int height) {
        return new Matrix(handle, data, height, data.length / height, height);
    }

    /**
     * The handle for this matrix.
     *
     * @return The handle for this matrix.
     */
    public Handle getHandle() {
        return handle;
    }

    /**
     * Batch vector vector dot product. This vector is set as the dot product of
     * a and b.
     *
     * @param timesAB Multiply this by the product of a and b.
     * @param a The first vector. A sub vector of a matrix or greater vector.
     * @param b The second vector. A sub vector of a matrix or greater vector.
     * @param timesThis multiply this before adding to it.
     */
    public void addBatchVecVecMult(double timesAB, VectorsStride a, VectorsStride b, double timesThis) {

        data.getAsBatch(inc(), 1, dim()).multMatMatStridedBatched(handle,
                false, true,
                1, a.getSubVecDim(), 1,
                timesAB,
                a.data, a.inc(),
                b.data, b.inc(),
                timesThis, inc()
        );
    }

    /**
     * Batch vector vector dot product. This vector is set as the dot product of
     * a and b.
     *
     * @param a The first vector. A sub vector of a matrix or greater vector.
     *
     * @param b The second vector. A sub vector of a matrix or greater vector.
     * @return this
     *
     */
    public Vector setBatchVecVecMult(VectorsStride a, VectorsStride b) {
        addBatchVecVecMult(1, a, b, 0);
        return this;
    }

    /**
     * Partitions this vector into a sets of incremental subsets.
     *
     * @param numParts The number of subsets.
     * @return An array of incremental subsets.
     */
    public Vector[] parition(int numParts) {
        Vector[] part = new Vector[numParts];
        Arrays.setAll(part, i -> getSubVector(i, dim() / numParts, numParts));
        return part;
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param transposeMat Should the matrix be transposed.
     * @param timesAB Gets multiplied by the product.
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @param timesCurrent Multiply this before adding the product.
     * @return The product is placed in this and this is returned.
     */
    public Vector addProduct(boolean transposeMat, double timesAB, Vector vec, Matrix mat, double timesCurrent) {
        data.multMatMat(handle,
                false, transposeMat,
                1, transposeMat ? mat.getHeight() : mat.getWidth(), vec.dim(),
                timesAB,
                vec.data, vec.inc(),
                mat.data, mat.colDist,
                timesCurrent, inc()
        );
        return this;
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @return The product is placed in this and this is returned.
     */
    public Vector setProduct(Vector vec, Matrix mat) {
        return addProduct(false, 1, vec, mat, 0);
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * This method does not work if increment does not equal 1. Try using matrix
     * methods instead or work with a vector that has an increment of 1.
     *
     * @param transposeMat Should the matrix be transposed.
     * @param timesAB Gets multiplied by the product.
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @param timesCurrent Multiply this before adding the product.
     * @return The product is placed in this and this is returned.
     */
    public Vector addProduct(boolean transposeMat, double timesAB, Matrix mat, Vector vec, double timesCurrent) {
//        if (inc() > 1) {
//            Matrix in = new Matrix(handle, data, dim(), 1);
//            DArray outTempStorageSite = data.subArray(dim());
//            Matrix outNeedsStorage = new Matrix(handle, data.subArray(1), colDist - 1, dim(), colDist);
//    //TODO: Finsih this section, otherwise this matrix has the wrong dimensions to recieve the product.
//    //be careful not to lose or overwirte any data.
//        }
//        
        return addProduct(!transposeMat, timesAB, vec, mat, timesCurrent);
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @return The product is placed in this and this is returned.
     */
    public Vector setProduct(Matrix mat, Vector vec) {
        return addProduct(false, 1, mat, vec, 0);
    }

    /**
     * The increment between elements of this vector. This is the column
     * distance of this matrix.
     *
     * @return The increment between elements of this vector.
     */
    public int inc() {
        return colDist;
    }

    /**
     * A set of vectors contained within this vector.
     *
     * @param stride The distance between the first elements of each vector.
     * @param batchSize The number of sub vectors.
     * @param subVectorDim The number of elements in each sub vector.
     * @param subVectorInc The increment of each sub vector over this vector.
     * @return The set of sub vectors.
     */
    public VectorsStride subVectors(int stride, int batchSize, int subVectorDim, int subVectorInc) {
        return new VectorsStride(
                handle,
                data,
                inc() * subVectorInc,
                subVectorDim,
                inc() * stride,
                batchSize
        );
    }

    /**
     * This vector as a double array.
     *
     * @return
     */
    public double[] vecGet() {
        return data.getIncremented(handle, inc());
    }

    /**
     * The L2norm or magnitude of this vector.
     *
     * @return The norm of this vector.
     */
    public double norm() {
        return data.norm(handle, dim(), colDist);
    }

    @Override
    public String toString() {
        return Arrays.toString(toArray());
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                DArray array = new DArray(hand, 1, 2, 3, 4, 5, 6);
                DArray a2 = new DArray(hand, 2, 2, 2, 2, 2, 2)) {
            Vector v = new Vector(hand, array, 1);
            v.ebeDivide(new Vector(hand, a2, 1));
            System.out.println(v);
        }
    }

}
