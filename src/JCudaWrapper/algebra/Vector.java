package JCudaWrapper.algebra;

import java.util.Arrays;
import java.util.Iterator;
import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.exception.OutOfRangeException;
import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.exception.MathArithmeticException;
import org.apache.commons.math3.exception.MathUnsupportedOperationException;
import org.apache.commons.math3.exception.NotPositiveException;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;
import org.apache.commons.math3.linear.RealVectorPreservingVisitor;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DSingleton;
import JCudaWrapper.array.DStrideArray;

/**
 * The {@code Vector} class extends {@code RealVector} and represents a vector
 * stored on the GPU. It relies on the {@code DArray} class for data storage and
 * the {@code Handle} class for JCublas operations.
 */
public class Vector extends RealVector implements AutoCloseable {

    private final DArray data;  // Underlying array for GPU-based operations
    private final Handle handle; // JCublas handle for GPU operations
    public final int inc;

    /**
     * Constructs a new {@code Vector} from an existing data pointer on the GPU.
     *
     * @param data The {@code DArray} storing the vector on the GPU.
     * @param inc The increment between elements of the data that make of this
     * vector.
     * @param handle The JCublas handle for GPU operations.
     */
    public Vector(Handle handle, DArray data, int inc) {
        this.data = data;
        this.handle = handle;
        this.inc = inc;
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
     * {@inheritDoc}
     */
    @Override
    public double getEntry(int index) throws OutOfRangeException {
        checkIndex(index);
        return data.get(index * inc).getVal(handle);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setEntry(int index, double value) throws OutOfRangeException {
        checkIndex(index);
        data.set(handle, index * inc, value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getDimension() {
        return Math.ceilDiv(data.length, inc);
    }

    /**
     * Adds another vector times a scalar to this vector, changing this vector.
     *
     * @param mult A scalar to be multiplied by @code{v} before adding it to
     * this vector.
     * @param v The vector to be added to this vector.
     * @return This vector.
     */
    public Vector addToMe(double mult, Vector v) {
        checkVectorLength(v);

        data.addToMe(handle, mult, v.data, v.inc, inc);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapMultiplyToSelf(double scalar) {
        data.multMe(handle, scalar, inc);
        return this;
    }

    /**
     * Sets all the values in this vector to that of the scalar.
     *
     * @param scalar The new value to fill this vector.
     * @return This vector.
     */
    public Vector fill(double scalar) {
        if (scalar == 0 && inc == 1) {
            data.fill0(handle);
        } else {
            data.fill(handle, scalar, inc);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector add(RealVector v) throws DimensionMismatchException {

        try (Vector other = new Vector(handle, v.toArray())) {
            return add(other);
        }
    }

    /**
     * {@inheritDoc}
     */
    public Vector add(Vector v) throws DimensionMismatchException {

        return copy().addToMe(1, v);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapMultiply(double d) {

        return copy().mapMultiplyToSelf(d);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double dotProduct(RealVector v) throws DimensionMismatchException {

        try (Vector temp = new Vector(handle, v.toArray())) {
            return dotProduct(temp);
        }

    }

    /**
     * Computes the dot product of this vector with another vector.
     *
     * @see Vector#dotProduct(org.apache.commons.math3.linear.RealVector)
     * @param v The other vector to compute the dot product with.
     * @return The dot product of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public double dotProduct(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);

        return data.dot(handle, v.data, v.inc, inc);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public RealVector subtract(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return subtract(temp);
        }
    }

    /**
     * Subtracts another vector from this vector element-wise.
     *
     * @see Vector#subtract(org.apache.commons.math3.linear.RealVector)
     * @param v The vector to subtract.
     * @return A new vector that is the difference of this vector and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector subtract(Vector v) throws DimensionMismatchException {

        return copy().addToMe(-1, v);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector copy() {
        if (inc == 1) {
            return new Vector(handle, data.copy(handle), inc);
        }
        Vector copy = new Vector(handle, getDimension());
        copy.data.set(handle, data, 0, 0, 1, inc, getDimension());
        return copy;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector ebeMultiply(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return ebeMultiply(temp);
        }
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
    public Vector mapEbeMultiplyToSelf(Vector a, Vector b) {

        return mapAddEbeMultiplyToSelf(a, b, 0);

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
    public Vector mapAddEbeMultiplyToSelf(Vector a, Vector b, double timesThis) {
        return addEbeMultiplyToSelf(1, a, b, timesThis);
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
        checkVectorLength(a, b);

        data.multSymBandMatVec(handle, true,
                getDimension(), 0,
                timesAB, a.data, a.inc,
                b.data, b.inc,
                timesThis, inc
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
     * @param b The second vector.
     * @param timesAB A scalar to multiply by a and b.
     * @param timesThis multiply this vector before adding the product of a and
     * b.
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector mapAddEbeMultiplyToSelf(Vector workSpace, double timesAB, double timesThis, Vector... a) {

        mapMultiplyToSelf(timesThis);

        if (a.length == 0) {
            return this;
        }
        if (a.length == 1) {
            return addEbeMultiplyToSelf(timesAB, this, a[0], 1);
        }
        if (a.length == 2) {
            return addEbeMultiplyToSelf(timesAB, a[0], a[1], 1);
        }

        workSpace.addEbeMultiplyToSelf(timesAB, a[0], a[1], 0);

        for (int i = 2; i < a.length; i++) {
            workSpace.mapAddEbeMultiplyToSelf(workSpace, a[i], 0);
        }

        return addToMe(1, workSpace);

    }

    /**
     * Computes the element-wise product of this vector and another vector.
     *
     * @see Vector#ebeMultiply(org.apache.commons.math3.linear.RealVector)
     * @param v The other vector.
     * @return A new vector containing the element-wise product of this vector
     * and {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeMultiply(Vector v) {
        checkVectorLength(v);

        return new Vector(handle, data.length).mapEbeMultiplyToSelf(this, v);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector ebeDivide(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return ebeDivide(temp);
        }
    }

    /**
     * A vector with every element raised to -1.
     *
     * @return A new vector with every element raised to -1.
     */
    public Vector ebeInvert() {

        return mapEBEInverse(new Vector(handle, getDimension()));
    }

    /**
     * Maps the inverse of each element in this vectot to the target. DO NOT
     * pass this into mapTo. DO NOT use this method to map to itself.
     *
     * @param mapTo Where the inverse of the elements in this vector are to be
     * put. This may not be this vector or have any overlapping data location
     * with this vector.
     * @return The vector mapTo with its elements overwritten to be the inverse
     * of the elements in this vector.
     */
    public Vector mapEBEInverse(Vector mapTo) {
        mapTo.fill(1);

        mapTo.data.solveTriangularBandedSystem(
                handle, true, false, false,
                getDimension(), 0, data, 1, 1);

        return mapTo;
    }

    /**
     * Maps the inverse of each element in this vectot to the target. DO NOT
     * pass this into mapTo. DO NOT use this method to map to itself.
     *
     * @param numerator The the numerator. The elements in this vector become
     * the denominator, and the result is stored in numerator.
     * @return The vector numerator, now divided by this.
     */
    public Vector mapEBEDivide(Vector numerator) {

        numerator.data.solveTriangularBandedSystem(
                handle, true, false, false,
                getDimension(), 0, data, 1, 1);

        return numerator;
    }

    /**
     * Computes the element-wise division of this vector by another vector.
     *
     * @see Vector#ebeDivide(org.apache.commons.math3.linear.RealVector)
     * @param v The vector to divide by.
     * @return A new vector containing the element-wise division of this vector
     * by {@code v}.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    public Vector ebeDivide(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);

        try (Vector inverse = ebeInvert()) {

            return ebeMultiply(inverse);
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getNorm() {
        return data.norm(handle, getDimension(), inc);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void checkIndex(int index) {
        if (index < 0 || index >= getDimension()) {
            throw new OutOfRangeException(index, 0, getDimension() - 1);
        }
    }

    /**
     * Checks if the dimensions of this vector match the dimensions of the given
     * vector.
     *
     * @param v The vector to compare dimensions with.
     * @throws DimensionMismatchException if the vectors have different lengths.
     */
    private void checkVectorLength(RealVector... vec) {
        for (RealVector v : vec) {
            if (v.getDimension() != getDimension()) {
                throw new DimensionMismatchException(v.getDimension(), getDimension());
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector append(RealVector rv) {
        try (Vector temp = new Vector(handle, rv.toArray())) {
            return append(temp);
        }
    }

    /**
     * A new vector that is the concatenation of this vector and given one. This
     * vector does not change.
     *
     * @see Vector#append(org.apache.commons.math3.linear.RealVector)
     * @param rv The vector to be concatenated to this one.
     * @return A The concatenation of the two vectors.
     */
    public Vector append(Vector rv) {
        DArray append = DArray.empty(getDimension() + rv.getDimension());

        append.set(handle, data, 0, 0, 1, inc, getDimension());
        append.set(handle, rv.data, getDimension(), 0, 1, rv.inc, rv.getDimension());
        return new Vector(handle, append, 1);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector append(double d) {
        try (DSingleton toAppend = new DSingleton(handle, d)) {
            return append(new Vector(handle, toAppend, 1));
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
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
                data.subArray(begin * inc, inc * increment * (length - 1) + 1),
                inc * increment
        );
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setSubVector(int i, RealVector rv) throws OutOfRangeException {
        data.set(handle, rv.toArray(), i * inc, 0, rv.getDimension());
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
        data.set(handle, rv.data, i * inc, 0, inc, rv.inc, rv.getDimension());
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
        for (int mCol = 0; mCol < getDimension(); mCol++) {
            setSubVector(toIndex + mCol * m.getHeight(), m.getColumnVector(mCol));
        }
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
    public void set(Matrix m) {
        for (int mCol = 0; mCol < getDimension(); mCol++) {
            setSubVector(mCol * m.getHeight(), m.getColumnVector(mCol));
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
     * {@inheritDoc}
     */
    @Override
    public boolean isNaN() {
        return Double.isNaN(dotProduct(this));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isInfinite() {
        return Double.isInfinite(dotProduct(this));
    }

    /**
     * Closes the underlying memory, so be sure it's not being used elsewhere.
     */
    @Override
    public void close() {
        if (inc != 1) {
            throw new IllegalAccessError("You are cleaning data from a sub vector");
        }
        data.close();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String toString() {
        return Arrays.toString(toArray());
    }

    /**
     * Compares this vector to another vector and checks if they are equal
     * within a specified tolerance.
     *
     * This method subtracts the given vector from this vector and compares the
     * norm of the resulting vector to the specified epsilon. If the norm is
     * less than epsilon, the vectors are considered equal, indicating that the
     * difference between them is smaller than the given tolerance.
     *
     * @param other The vector to compare with this vector.
     * @param epsilon The tolerance value within which the vectors are
     * considered equal. Must be a non-negative number.
     * @return true if the difference between the vectors is less than the
     * specified epsilon, false otherwise. *
     */
    public boolean equals(Vector other, double epsilon) {
        if (other.getDimension() != getDimension()) {
            return false;
        }
        return subtract(other).getNorm() < epsilon;
    }

    /**
     * Compares this vector to another vector and checks if they are equal
     * within a default tolerance of 1e-10.
     *
     * This method subtracts the given vector from this vector and compares the
     * norm of the resulting vector to the specified epsilon. If the norm is
     * less than epsilon, the vectors are considered equal, indicating that the
     * difference between them is smaller than the given tolerance.
     *
     * @param other The vector to compare with this vector.
     * @return true if the difference between the vectors is less than the
     * specified epsilon, false otherwise. *
     */
    public boolean equals(Vector other) throws MathUnsupportedOperationException {
        return equals(other, 1e-10);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void addToEntry(int index, double increment) throws OutOfRangeException {
        checkIndex(index);
        data.set(handle, index * inc, getEntry(index) + increment);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapAdd(double d) {
        return copy().mapAddToSelf(d);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapAddToSelf(double d) {
        try (DSingleton toAdd = new DSingleton(handle, d)) {
            data.addToMe(handle, 1, toAdd, 0, inc);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double cosine(RealVector v) throws DimensionMismatchException, MathArithmeticException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return cosine(temp);
        }
    }

    /**
     * Computes the cosine of the angle between this vector and the argument.
     *
     * @see Vector#cosine(org.apache.commons.math3.linear.RealVector)
     * @param other Vector
     * @return the cosine of the angle between this vector and v.
     */
    public double cosine(Vector other) {
        checkVectorLength(other);

        double[] norms = new double[2];
        double dot;
        try (Handle aNormHand = new Handle(); Handle bNormHand = new Handle()) {
            data.norm(aNormHand, getDimension(), inc, norms, 0);
            other.data.norm(bNormHand, other.getDimension(), other.inc, norms, 1);
            dot = dotProduct(other);
        }

        return dot / (norms[0] * norms[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getDistance(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return getDistance(temp);
        }
    }

    /**
     * {@inheritDoc}
     */
    public double getDistance(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);
        try (Vector diff = subtract(v)) {
            return diff.getNorm();
        }

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getL1Norm() {
        return data.sumAbs(handle, getDimension(), inc);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getLInfNorm() {
        return getEntry(data.argMaxAbs(handle, getDimension(), inc));
    }

    /**
     * {@inheritDoc}
     */
    public double getL1Distance(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);
        try (Vector diff = subtract(v)) {
            return diff.getL1Norm();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getL1Distance(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return getL1Distance(temp);
        }
    }

    /**
     * {@inheritDoc}
     */
    public double getLInfDistance(Vector v) throws DimensionMismatchException {
        checkVectorLength(v);
        try (Vector diff = subtract(v)) {
            return diff.getLInfNorm();
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getLInfDistance(RealVector v) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return getLInfDistance(temp);
        }
    }

    /**
     * Finds the index of the minimum or maximum element of the vector.
     *
     * @param isMax True to find the argMaximum, false for the argMin.
     * @return The argMin or argMax.
     */
    private int getMinMaxInd(boolean isMax) {
        int argMaxAbsVal = data.argMaxAbs(handle, getDimension(), inc);
        double maxAbsVal = getEntry(argMaxAbsVal);
        if (maxAbsVal == 0) {
            return 0;
        }
        if (maxAbsVal > 0 && isMax) {
            return argMaxAbsVal;
        }
        if (maxAbsVal < 0 && !isMax) {
            return argMaxAbsVal;
        }

        try (Vector sameSign = mapAdd(maxAbsVal)) {
            return sameSign.data.argMinAbs(handle, getDimension(), inc);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getMinIndex() {
        return getMinMaxInd(false);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getMinValue() {
        return getEntry(getMinIndex());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getMaxIndex() {
        return getMinMaxInd(true);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double getMaxValue() {
        return getEntry(getMaxIndex());
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapSubtract(double d) {
        return copy().mapSubtractToSelf(d);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapSubtractToSelf(double d) {
        try (DSingleton add = new DSingleton(handle, d)) {
            data.addToMe(handle, -1, add, 0, inc);
        }
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapDivideToSelf(double d) {
        mapMultiplyToSelf(1 / d);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector mapDivide(double d) {
        return mapMultiply(1 / d);
    }

    /**
     * @param v The vector with which this one is creating an outer product.
     * @return The outer product.
     * @see Vector#outerProduct(org.apache.commons.math3.linear.RealVector)
     */
    public Matrix outerProduct(Vector v) {
        Matrix outerProd = new Matrix(handle, getDimension(), v.getDimension()).fill(0);
        outerProd.dArray().outerProd(handle, getDimension(), v.getDimension(), 1, data, inc, v.data, v.inc, getDimension());
        return outerProd;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Matrix outerProduct(RealVector v) {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return outerProduct(temp);
        }
    }

    /**
     * @see Vector#projection(org.apache.commons.math3.linear.RealVector)
     */
    public Vector projection(Vector v) throws DimensionMismatchException, MathArithmeticException {
        double[] dots = new double[2];

        try (Handle dotHandle = new Handle()) {
            data.dot(dotHandle, v.data, v.inc, inc, dots, 0);
            v.data.dot(handle, v.data, v.inc, inc, dots, 1);
        }
        return v.mapMultiplyToSelf(dots[0] / dots[1]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector projection(RealVector v) throws DimensionMismatchException, MathArithmeticException {
        try (Vector temp = new Vector(handle, v.toArray())) {
            return projection(temp);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void set(double value) {
        fill(value);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public double[] toArray() {
        if (inc != 1)
            try (Vector copy = copy()) {
            return copy.data.get(handle);
        }
        return data.get(handle);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector unitVector() throws MathArithmeticException {
        Handle normHand = new Handle();
        double[] norm = new double[1];
        data.norm(normHand, getDimension(), inc, norm, 0);
        Vector copy = copy();
        normHand.close();
        return copy.mapMultiplyToSelf(1 / norm[0]);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void unitize() throws MathArithmeticException {
        mapMultiplyToSelf(1 / getNorm());
    }

    /**
     * Provides an array of entries, one for each underlying element.
     *
     * @return An entry for each element.
     */
    private Entry[] getEntries() {
        double[] toArray = toArray();
        Entry[] entries = new Entry[toArray.length];
        Arrays.setAll(entries, i -> new Entry() {
            int ind = i;

            @Override
            public double getValue() {
                return toArray[ind];
            }

            @Override
            public void setValue(double value) {
                setEntry(ind, value);
                toArray[ind] = value;
            }

            @Override
            public int getIndex() {
                return ind;
            }

            @Override
            public void setIndex(int index) {
                ind = index;
            }
        });
        return entries;
    }

    /**
     * Note, this method is slow and does not use the gpu.
     *
     * {@inheritDoc}
     */
    @Override
    public Iterator<Entry> sparseIterator() {

        Entry[] entries = getEntries();

        return new Iterator<RealVector.Entry>() {
            int i = 0;

            @Override
            public boolean hasNext() {
                return i < entries.length;
            }

            @Override
            public Entry next() {
                Entry next = entries[i];
                i++;
                while (i < entries.length && entries[i].getValue() == 0) {
                    i++;
                }
                return next;
            }
        };
    }

    /**
     * Note, this method is slow and does not use the gpu.
     *
     * {@inheritDoc}
     */
    @Override
    public Iterator<Entry> iterator() {

        Entry[] entries = getEntries();

        return new Iterator<RealVector.Entry>() {
            int i = 0;

            @Override
            public boolean hasNext() {
                return i < entries.length;
            }

            @Override
            public Entry next() {
                Entry next = entries[i];
                i++;
                return next;
            }
        };
    }

    /**
     * Note, this method is slow and does not use the gpu.
     *
     * {@inheritDoc}
     */
    @Override
    public Vector map(UnivariateFunction function) {
        double[] toArray = toArray();
        double[] mapped = new double[getDimension()];
        Arrays.setAll(mapped, i -> function.value(toArray[i]));
        return new Vector(handle, mapped);
    }

    /**
     * Note, this method is slow and does not use the gpu. {@inheritDoc}
     */
    @Override
    public Vector mapToSelf(UnivariateFunction function) {
        try (Vector map = map(function)) {
            data.set(handle, map.data, 0, 0, inc, 1, getDimension());
            return this;
        }
    }

    /**
     * @see Vector#combineToSelf(double, double,
     * org.apache.commons.math3.linear.RealVector)
     */
    public Vector combineToSelf(double a, double b, Vector y) throws DimensionMismatchException {
        mapMultiplyToSelf(a);
        addToMe(b, y);
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector combineToSelf(double a, double b, RealVector y) throws DimensionMismatchException {
        try (Vector temp = new Vector(handle, y.toArray())) {
            return combineToSelf(a, b, temp);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Vector combine(double a, double b, RealVector y) throws DimensionMismatchException {
        return copy().combine(a, b, y);
    }

    /**
     * {@inheritDoc} Walks through the elements of the vector, invoking the
     * visitor's start method, visiting each element, and finally invoking the
     * end method.
     *
     * @param walker the visitor that is passed to each element
     */
    @Override
    public double walkInDefaultOrder(RealVectorChangingVisitor walker) {

        double[] dataArray = toArray();

        walker.start(getDimension(), 0, getDimension() - 1);

        for (int i = 0; i < getDimension(); i++) {
            walker.visit(i, dataArray[i]);
        }

        if (inc == 1) {
            data.set(handle, dataArray);
        } else {
            try (DArray temp = new DArray(handle, dataArray)) {
                data.set(handle, temp, 0, 0, inc, 1, getDimension());
            }
        }

        return walker.end();
    }

    /**
     * {@inheritDoc} Walks through the elements of the vector in optimized
     * order, invoking the visitor's start method, visiting each element, and
     * finally invoking the end method.
     *
     * @param walker the visitor that is passed to each element
     * @param start the start index
     * @param end the end index
     */
    @Override
    public double walkInOptimizedOrder(RealVectorChangingVisitor walker, int start, int end) {

        double[] dataArray = toArray();

        walker.start(getDimension(), start, end);

        if (start <= end) {
            for (int i = start; i <= end; i++) {
                walker.visit(i, dataArray[i]);
            }
        } else {
            for (int i = start; i >= end; i--) {
                walker.visit(i, dataArray[i]);
            }
        }

        if (inc == 1) {
            data.set(handle, dataArray);
        } else try (DArray temp = new DArray(handle, dataArray)) {
            data.set(handle, temp, 0, 0, inc, 1, getDimension());
        }

        return walker.end();
    }

    /**
     * {@inheritDoc} Walks through the elements of the vector in default order,
     * invoking the visitor's start method, visiting each element. Does not
     * modify the data.
     *
     * @param walker the visitor that is passed to each element
     */
    @Override
    public double walkInDefaultOrder(RealVectorPreservingVisitor walker) {

        double[] dataArray = toArray();

        walker.start(getDimension(), 0, getDimension() - 1);

        for (int i = 0; i < getDimension(); i++) {
            walker.visit(i, dataArray[i]);
        }

        return walker.end();
    }

    /**
     * {@inheritDoc} Walks through the elements of the vector in optimized
     * order, invoking the visitor's start method, visiting each element. Does
     * not modify the data.
     *
     * @param walker the visitor that is passed to each element
     * @param start the start index
     * @param end the end index
     */
    public double walkInOptimizedOrder(RealVectorPreservingVisitor walker, int start, int end) {

        // Retrieve data from GPU
        double[] dataArray = data.get(handle);

        // Start the walk
        walker.start(getDimension(), start, end);

        // Visit elements in the specified order
        if (start <= end) {
            for (int i = start; i <= end; i++) {
                walker.visit(i, dataArray[i]);
            }
        } else {
            for (int i = start; i >= end; i--) {
                walker.visit(i, dataArray[i]);
            }
        }

        // Complete the walk
        return walker.end();
    }

    /**
     * Creates a vertical matrix from this vector. Note, this method only works
     * if the stride increment is 1. Otherwise an exception is thrown.
     *
     * If a vertical matrix is desired for a stride increment that's not one,
     * create a horizontal matrix and transpose it.
     *
     * @return A vertical matrix representing this vector.
     */
    public Matrix vertical() {
        if (inc != 1) {
            throw new RuntimeException("The stride increment must be one.");
        }
        return new Matrix(handle, data, getDimension(), 1);
    }

    /**
     * Creates a horizontal matrix from this vector.
     *
     * @return A horizontal matrix from this vector.
     */
    public Matrix horizontal() {
        return new Matrix(handle, data, 1, getDimension(), inc);
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

        data.getAsBatch(inc, getDimension(), 1).multMatMatStridedBatched(handle,
                false, true,
                1, a.getSubVecDim(), 1,
                timesAB,
                a.data, a.inc,
                b.data, b.inc,
                timesThis, inc
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
        Arrays.setAll(part, i -> getSubVector(i, getDimension() / numParts, numParts));
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
    public Vector multiplyAndAdd(boolean transposeMat, double timesAB, Vector vec, Matrix mat, double timesCurrent) {
        data.multMatMat(handle,
                false, transposeMat,
                1, transposeMat ? mat.getHeight() : mat.getWidth(), vec.getDimension(),
                timesAB,
                vec.data, vec.inc,
                mat.data, mat.colDist,
                timesCurrent, inc
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
    public Vector multiplyAndSet(Vector vec, Matrix mat) {
        return multiplyAndAdd(false, 1, vec, mat, 0);
    }

    /**
     * Multiplies the vector and the matrix and places the product here.
     *
     * This method does not work if increment does not equal 1.  Try using matrix
     * methods instead or work with a vector that has an increment of 1.
     * 
     * @param transposeMat Should the matrix be transposed.
     * @param timesAB Gets multiplied by the product.
     * @param vec The vector to be multiplied.
     * @param mat The matrix to be multiplied.
     * @param timesCurrent Multiply this before adding the product.
     * @return The product is placed in this and this is returned.
     */
    public Vector multiplyAndAdd(boolean transposeMat, double timesAB, Matrix mat, Vector vec, double timesCurrent) {
        if(inc != 1) throw new IllegalArgumentException("Unforutunately this method does not work if the increment does not equal 1.");
        data.multMatMat(handle,
                transposeMat, true,
                transposeMat ? mat.getWidth() : mat.getHeight(),
                1,
                transposeMat ? mat.getHeight() : mat.getWidth(),
                timesAB,
                mat.data, mat.colDist,
                vec.data, vec.inc,
                timesCurrent, getDimension()
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
    public Vector multiplyAndSet(Matrix mat, Vector vec) {
        return multiplyAndAdd(false, 1, mat, vec, 0);
    }

    /**
     * A set of vectors contained within this vector.
     * @param stride The distance between the first elements of each vector.
     * @param batchSize The number of sub vectors.
     * @param subVectorDim The number of elements in each sub vector.
     * @param subVectorInc The increment of each sub vector over this vector.
     * @return The set of sub vectors.
     */
    public VectorsStride subVectors(int stride, int batchSize, int subVectorDim, int subVectorInc){
        
        return new VectorsStride(
                handle, 
                data.getAsBatch(stride*inc, batchSize, subVectorDim*inc*subVectorInc), 
                inc*subVectorInc
        );
    }
    
    
    
    
    public static void main(String[] args) {
        try(Handle hand = new Handle(); DArray array = new DArray(hand, 1,2,3,4,5,6); DArray a = new DArray(hand, 1, 1); DArray result = DArray.empty(3)){
            Matrix mat = new Matrix(hand, array, 3, 2);
            Vector vec = new Vector(hand, a, 1);
            Vector resultVec = new Vector(hand, result, 1);
                        
            System.out.println("mat = \n" + mat);
            System.out.println("vec = \n" + vec);
            
            
            resultVec.multiplyAndSet(mat, vec);
            
            System.out.println("result = \n" + resultVec);
        }
    }
    
    
}
