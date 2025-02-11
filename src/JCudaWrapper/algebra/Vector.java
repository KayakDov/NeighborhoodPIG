package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray1d;
import JCudaWrapper.array.DArray2d;
import java.util.Arrays;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.exception.NotPositiveException;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.DSingleton;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;

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
    public Vector(Handle handle, DArray1d data, int inc) {
        super(handle, data.as2d(inc));
    }

    /**
     * Constructs a new {@code Vector} from a 1D array.
     *
     * @param array The array storing the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, double... array) {
        this(handle, new DArray1d(array.length).set(handle, array), 1);
    }

    /**
     * Constructs a new empty {@code Vector} of specified length.
     *
     * @param length The length of the vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, int length) {
        this(handle, new DArray1d(length), 1);
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
        data.get(index).set(handle, value);
    }

    /**
     * The dimension of the vector. The number of elements in it.
     *
     * @return The dimension of the vector. The number of elements in it.
     */
    public int dim() {
        return inc() == 0 ? 1 : (int) Math.ceil((double) data.size() / inc());
    }

    /**
     * Adds another vector times a scalar to this vector, changing this vector.
     *
     * @param mult A scalar to be multiplied by @code{v} before adding it to
     * this vector.
     * @param v The vector to be added to this vector.
     * @return This vector.
     */
    public Vector add(double mult, Vector v) {
        data.as1d().add(handle, mult, v.data.as1d(), v.inc(), inc());
        return this;
    }

    /**
     * Adds the scalar to every element in this vector.
     *
     * @param scalar To be added to every element in this vector.
     * @return this.
     */
    @Override
    public Vector add(double scalar) {
        try (DSingleton oneOne = new DSingleton().set(handle, scalar)) {
            data.as1d().add(handle, 1, oneOne, 0, inc());
        }
        return this;

    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public double dotProduct(Vector v) {

        return data.as1d().dot(handle, v.data, v.inc(), inc());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector copy() {
        if (inc() == 1) {
            return new Vector(handle, data.copy(handle).as1d(), inc());
        }
        Vector copy = new Vector(handle, dim());
        copy.data.set(handle, data, inc(), 1);
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
    public Vector ebeSetProduct(Vector a, Vector b) {

        return ebeAddProduct(a, b, 0);

    }

    /**
     * multiplies this vector by the given scalar and vector (element by
     * element).
     *
     * @param scalar The scalar times this.
     * @param a The vector that will be ebe times this.
     * @return this.
     */
    public Vector multiply(double scalar, Vector a) {
        return addEbeProduct(scalar, a, this, 0);
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
    public Vector ebeAddProduct(Vector a, Vector b, double timesThis) {
        return addEbeProduct(1, a, b, timesThis);
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
    public Vector addEbeProduct(double timesAB, Vector a, Vector b, double timesThis) {

        Kernel.run("addEBEProduct",
                handle,
                dim(),
                data,
                P.to(inc()),
                P.to(timesAB),
                P.to(a),
                P.to(a.inc()),
                P.to(b),
                P.to(b.inc()),
                P.to(timesThis)
        );

        return this;
    }

    /**
     * Element by element division. Like most methods, it changes this vector.
     *
     * @param denominator the denominator.
     * @return this
     */
    public Vector ebeDivide(Vector denominator) {

        addEBEDivide(1, this, denominator, 0);
        return this;
    }
    
    /**
     * Adds the quotient to this method.
     * @param timesQuotient
     * @param numerator The numerator of the quotient.
     * @param denominator The denominator of the quotient.
     * @param timesThis Times this before anything is added.
     * @return this.
     */
    public Vector addEBEDivide(double timesQuotient, Vector numerator, Vector denominator, double timesThis) {

        Kernel.run("addEBEQuotient",
                handle,
                dim(),
                data,
                P.to(inc()),
                P.to(timesQuotient),
                P.to(numerator),
                P.to(numerator.inc()),
                P.to(denominator),
                P.to(denominator.inc()),
                P.to(timesThis)
        );

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
        return subVector(begin, length, 1);
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
    public Vector subVector(int begin, int length, int increment) throws NotPositiveException, OutOfRangeException {
        return new Vector(
                handle,
                data.as1d().sub(begin * inc(), inc() * increment * (length - 1) + 1),
                inc() * increment
        );
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
            setSubVector(toIndex + mCol * m.height(), m.getColumn(mCol));
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
        workSpace.setSum(1, v, -1, this);
        return workSpace.norm();
    }

    /**
     * {@inheritDoc}
     *
     * @param alpha
     * @param a
     * @param beta
     * @param b
     * @return
     */
    @Override
    public Vector setSum(double alpha, Matrix a, double beta, Matrix b) {
        super.setSum(alpha, a, beta, b);
        return this;
    }

    /**
     * The L_1 norm.
     */
    public double getL1Norm() {
        return data.as1d().sumAbs(handle, dim(), inc());
    }

    /**
     * The L_infinity norm
     */
    public double getLInfNorm() {
        return get(data.as1d().argMaxAbs(handle, dim(), inc()));
    }

    /**
     * Finds the index of the minimum or maximum element of the vector. This
     * method creates its own workspace equal in size to this.
     *
     * @param isMax True to find the argMaximum, false for the argMin.
     * @return The argMin or argMax.
     */
    private int getMinMaxInd(boolean isMax) {
        int argMaxAbsVal = data.as1d().argMaxAbs(handle, dim(), inc());
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

        try (Vector sameSign = copy().add(maxAbsVal)) {
            return sameSign.data.as1d().argMinAbs(handle, dim(), inc());
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
    public Matrix outerProduct(Vector v, DArray2d placeOuterProduct) {
        placeOuterProduct.outerProd(handle, 1, data.as1d(), inc(), v.data.as1d(), v.inc());
        return new Matrix(handle, placeOuterProduct);
    }

    /**
     * @see Vector#projection(org.apache.commons.math3.linear.RealVector)
     *
     * @param v project onto.
     * @return The projection.
     *
     */
    public Vector projection(Vector v){
        double[] dots = new double[2];

        dots[0] = dotProduct(v);
        dots[1] = v.dotProduct(v);

        return (Vector)v.multiply(dots[0] / dots[1]);
    }

    /**
     * The cpu array that is a copy of this gpu vector.
     *
     * @return the array in the cpu.
     */
    public double[] toArray() {
        return data.get(handle, inc());
    }

    /**
     * Turn this vector into a unit vector.
     */
    public void unitize() throws MathArithmeticException {
        Vector.this.multiply(1 / norm());
    }

    /**
     * The data underlying this vector.
     *
     * @return The underlying data from this vector.
     */
    public DArray2d array() {
        return data;
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
     * Partitions this vector into a sets of incremental subsets.
     *
     * @param numParts The number of subsets.
     * @return An array of incremental subsets.
     */
    public Vector[] parition(int numParts) {
        Vector[] part = new Vector[numParts];
        Arrays.setAll(part, i -> subVector(i, dim() / numParts, numParts));
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
        data.addProduct(handle,
                false, transposeMat,
                1, transposeMat ? mat.height() : mat.width(), vec.dim(),
                timesAB,
                vec.data, vec.inc(),
                mat.data, mat.data.ld(),
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

        data.addProduct(handle, transposeMat,
                mat.height(), mat.width(),
                timesAB, mat.data, mat.colDist(),
                vec.array(), vec.inc(),
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
        return data.ld();
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
    public VectorsStride subVectors(int stride, int subVectorDim, int subVectorInc, int batchSize) {
        return new VectorsStride(
                handle,
                data,
                inc() * subVectorInc,
                subVectorDim,
                inc() * stride,
                batchSize
        );
    }

    @Override
    public Matrix addProduct(boolean transposeA, boolean transposeB, double timesAB, Matrix a, Matrix b, double timesThis) {
        throw new UnsupportedOperationException("Use the addProduct methods that take vectors as parameters instead.");
    }


    /**
     * The L2norm or magnitude of this vector.
     *
     * @return The norm of this vector.
     */
    public double norm() {
        return data.as1d().norm(handle, dim(), inc());
    }

    @Override
    public String toString() {
        return Arrays.toString(toArray());
    }


}
