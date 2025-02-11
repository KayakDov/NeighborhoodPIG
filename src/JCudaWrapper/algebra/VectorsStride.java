package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import java.util.Arrays;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DStrideArray;

/**
 *
 * @author E. Dov Neimand
 */
public class VectorsStride extends MatricesStride {

    /**
     * The constructor.
     *
     * @param handle
     * @param data The underlying data.
     * @param subArrayInc The increment of each subvector.
     * @param dim The number of elements in each subvector.
     * @param strideSize The distance between the first elements of each
     * subevector.
     * @param batchSize The number of subvectors.
     */
    public VectorsStride(Handle handle, DArray data, int subArrayInc, int dim, int strideSize, int batchSize) {
        super(handle, data, 1, dim, subArrayInc, strideSize, batchSize);
    }

    /**
     * The constructor.
     *
     * @param handle
     * @param strideSize The stride size.
     * @param batchSize The number of vectors in the batch.
     * @param subVecDim The number of elements in each subvector.
     * @param inc The increment of each subvector.
     */
    public VectorsStride(Handle handle, int strideSize, int batchSize, int subVecDim, int inc) {
        this(handle,
                new DArray3d(DStrideArray.totalDataLength(strideSize, inc * subVecDim, batchSize)),
                inc,
                subVecDim,
                strideSize,
                batchSize
        );
    }

    /**
     * The element at the ith index in every subVector.
     *
     * @param i The index of the desired element.
     * @return The element at the ith index in every subVector.
     */
    public Vector getElement(int i) {
        return new Vector(
                handle,
                data.subArray(i * colDist),
                data.stride * colDist);
    }

    /**
     * Gets the subvector at the desired index.
     *
     * @param i The index of the desired subvector.
     * @return The subvector at the desired index.
     */
    public Vector getVector(int i) {
        return new Vector(handle, data.getBatchArray(i), colDist);

    }

    /**
     * The dimension of the subarrays.
     *
     * @return The number of elements in each sub array.
     */
    public int getSubVecDim() {
        return (int)Math.ceil((double)data.subArraySize/ colDist);
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param transposeMats Should the matrices be transposed.
     * @param mats The matrices.
     * @param vecs The vectors.
     * @param timesAB A scalar to multiply the product by.
     * @param timesThis A scalar to multiply this by before the product is add
     * here.
     */
    public VectorsStride addProduct(boolean transposeMats, double timesAB, MatricesStride mats, VectorsStride vecs, double timesThis) {
        //TODO: this method doesn't work because the reciecing vector, this, has the wrong dimensions.
        return addProduct(!transposeMats, timesAB, vecs, mats, timesThis);
        
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param transposeMats Should the matrices be transposed.
     * @param mats The matrices.
     * @param vecs The vectors.
     * @param timesAB A scalar to multiply the product by.
     * @param timesThis A scalar to multiply this by before the product is add
     * here.
     * @return this.
     */
    public VectorsStride addProduct(boolean transposeMats, double timesAB, VectorsStride vecs, MatricesStride mats, double timesThis) {
        super.addProduct(false, transposeMats, timesAB, vecs, mats, timesThis);
        return this;
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     * @return this
     */
    public VectorsStride setProduct(VectorsStride vecs, MatricesStride mats) {
        return addProduct(false, 1, vecs, mats, 0);
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     * @return this
     */
    public VectorsStride setProduct(MatricesStride mats, VectorsStride vecs) {
        addProduct(false, 1, mats, vecs, 0);
        return this;
    }

    /**
     * Partitions this into an array of vectors so that v[i].get(j) is the ith
     * element in the jth vector.
     *
     * @return Partitions this into an array of vectors so that v[i].get(j) is
     * the ith element in the jth vector.
     */
    public Vector[] vecPartition() {
        Vector[] parts = new Vector[getSubVecDim()];
        Arrays.setAll(parts, i -> elmntsAtVecInd(i));
        return parts;
    }

    /**
     * The element at the ith index of each subVector.
     *
     * @param i The index of the desired elements.
     * @return An array, a such that a_j is the ith element of the jth array.
     */
    public Vector elmntsAtVecInd(int i) {
        return new Vector(handle, data.subArray(i * inc(), (batchSize - 1)* strideSize + 1), data.stride);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public VectorsStride fill(double val) {
        super.fill(val);
        return this;
    }

    /**
     * @see MatricesStride#add(boolean, double, JCudaWrapper.algebra.MatricesStride, double, JCudaWrapper.array.DArray)
     */
    public VectorsStride add(boolean transpose, double timesToAdd, VectorsStride toAdd, double timesThis, DArray3d workSpace) {
        super.add(transpose, timesToAdd, toAdd, timesThis, workSpace);
        return this;
    }

    /**
     * A contiguous subset of the subvectors in this set.
     *
     * @param start The index of the first subvector.
     * @param length The number of subvectors.
     * @return The subset.
     */
    @Override
    public VectorsStride subBatch(int start, int length) {
        return new VectorsStride(
                handle,
                data.subBatch(start, length),
                inc(),
                dim(),
                data.stride,
                length
        );
    }

    /**
     * The data underlying these vectors.
     *
     * @return The data underlying these vectors.
     */
    public DStrideArray array() {
        return data;
    }

    /**
     * The increments between elements of the subvectors. This is the column
     * distance.
     *
     * @return The increments between elements of the subvectors. This is the
     * column distance.
     */
    public int inc() {
        return super.colDist(); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/OverriddenMethodBody
    }

    /**
     * Changes each element x to x squared.
     *
     * @param normsGoHere This array will hold the norm of each vector.
     * @return
     */
    public Vector norms(DArray3d normsGoHere) {
        Vector norms = new Vector(handle, normsGoHere, 1);
        norms.addBatchVecVecMult(1, this, this, 0);
        Kernel.run("sqrt", handle, norms.dim(), norms.array(), P.to(norms.inc()), P.to(norms), P.to(norms.inc()));
        return norms;
    }

    /**
     * Turns these vectors into unit vectors, and then multiplies them by the magnitude.
     *
     * @param magnitude The magnitude that each vector will be stretched of
     * squished to have.
     * @param workSpace Should be 2 * batchSize in length.
     * @return this.
     */
    public VectorsStride setVectorMagnitudes(double magnitude, DArray3d workSpace) {
        Vector norms = norms(workSpace.subArray(0, data.batchSize));

        Vector normsInverted = new Vector(handle, workSpace.subArray(data.batchSize), 1)
                .fill(magnitude).ebeDivide(norms);

        multiply(normsInverted);

        return this;
    }

    public static void main(String[] args) {
        try (Handle hand = new Handle();
                Vector vec = new Vector(hand, 1,2,3,4,99,  5,6,7,8,99,  9,10,11,12,99)) {
            
            int width = 3, height = 4, colDist = 5;
            
            VectorsStride rows = vec.subVectors(1, width, colDist, height);
            
            System.out.println(rows);//2,6,10
            
            
        }
    }

    /**
     * The vectors in this brought to the cpu.
     *    
     * @return each vector as a [row][column].
     */
    public double[][] copyToCPURows() {
        double[][] copy = new double[getBatchSize()][];

        Arrays.setAll(copy, i -> getVector(i).toArray());

        return copy;
    }

    @Override
    public String toString() {
        
        return Arrays.deepToString(copyToCPURows()).replace("],", "],\n");
    }

    /**
     * The number of elements in each vector. This is the width.
     *
     * @return The number of elements in each vector. This is the width.
     */
    public int dim() {
        return width;
    }
}
