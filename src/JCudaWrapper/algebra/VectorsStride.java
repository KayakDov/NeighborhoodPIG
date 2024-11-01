package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.KernelManager;
import java.util.Arrays;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class VectorsStride extends MatricesStride implements AutoCloseable {

    
    
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
        this(
                handle,
                DArray.empty(DStrideArray.minLength(strideSize, inc * subVecDim, batchSize)),
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
//        return getSubVector(data.stride*i)/inc, getSubVecDim());
    }

    /**
     * The dimension of the subarrays.
     *
     * @return The number of elements in each sub array.
     */
    public int getSubVecDim() {
        return Math.ceilDiv(data.subArrayLength, colDist);
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
    public void addProduct(boolean transposeMats, MatricesStride mats, VectorsStride vecs, double timesAB, double timesThis) {
        addProduct(!transposeMats, vecs, mats, timesAB, timesThis);
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
    public void addProduct(boolean transposeMats, VectorsStride vecs, MatricesStride mats, double timesAB, double timesThis) {
         super.addProduct(false, transposeMats, vecs, mats, timesAB, timesThis);        
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     * @return this
     */
    public VectorsStride setVecMatMult(VectorsStride vecs, MatricesStride mats) {
        addProduct(false, vecs, mats, 1, 0);
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
    public VectorsStride setMatVecMult(MatricesStride mats, VectorsStride vecs) {
        addProduct(false, mats, vecs, 1, 0);
        return this;
    }

    /**
     * Partitions this into an array of vectors so that v[i].get(j) is the ith
     * element in the jth vector.
     *
     * @return Partitions this into an array of vectors so that v[i].get(j) is
     * the ith element in the jth vector.
     */
    public Vector[] vecParition() {
        Vector[] parts = new Vector[getSubVecDim()];
        Arrays.setAll(parts, i -> get(i));
        return parts;
    }

    /**
     * The element at the ith index of each subVector.
     *
     * @param i The index of the desired elements.
     * @return An array, a such that a_j is the ith element of the jth array.
     */
    public Vector get(int i) {
        return new Vector(handle, data.subArray(i * colDist), data.stride);
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
     * @see MatricesStride#addToMe(boolean, double, JCudaWrapper.algebra.MatricesStride, double, JCudaWrapper.array.DArray) 
     */
    
    public VectorsStride addToMe(boolean transpose, double timesToAdd, VectorsStride toAdd, double timesThis, DArray workSpace) {
        super.addToMe(transpose, timesToAdd, toAdd, timesThis, workSpace);
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

    @Override
    public void close() {
        data.close();
    }

    /**
     * The data underlying these vectors.
     *
     * @return The data underlying these vectors.
     */
    public DStrideArray dArray() {
        return data;
    }
    
    
    /**
     * The increments between elements of the subvectors.  This is the column distance.
     * @return The increments between elements of the subvectors.  This is the column distance.
     */
    public int inc() {
        return super.getColDist(); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/OverriddenMethodBody
    }
    
    /**
     * Changes each element x to x squared.
     * @param normsGoHere This array will hold the norm of each vector.
     * @return 
     */
    public Vector norms(DArray normsGoHere){
        Vector norms = new Vector(handle, normsGoHere, 1);
        norms.addBatchVecVecMult(1, this, this, 0);
        KernelManager.get("sqrt").mapToSelf(handle, norms);
        return norms;
    }
    
    /**
     * Turns these vectors into unit vectors.
     * @param magnitude The magnitude that each vector will be stretched of squished to have.
     * @param workSpace Should be 2 * batchSize in length.
     * @return this.
     */
    public VectorsStride setVectorMagnitudes(double magnitude, DArray workSpace){
        Vector norms = norms(workSpace.subArray(0, data.batchSize));        
        
        Vector normsInverted = new Vector(handle, workSpace.subArray(data.batchSize), 1)
                .fill(magnitude).ebeDivide(norms);
        
        multMe(normsInverted);
        
        return this;
    }
    
    /**
     * The vectors in this brought to the cpu.
     * @param workspace should be dim in length.
     * @return each vector as a [row][column].
     */
    public double[][] copyToCPURows(DArray workspace){
        double[][] copy = new double[height][];
        
        Arrays.setAll(copy, i -> getVector(i).toArray(workspace));
        
        return copy;
    }
    /**
     * The number of elements in each vector.  This is the width.
     * @return The number of elements in each vector.  This is the width.
     */
    public int dim(){
        return width;
    }
}
