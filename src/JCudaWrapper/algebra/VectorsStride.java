package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import java.util.Arrays;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class VectorsStride extends Vector {

    public final DStrideArray data;

    /**
     * The constructor.
     *
     * @param handle
     * @param data
     * @param inc The increment of each subvector.
     */
    public VectorsStride(Handle handle, DStrideArray data, int inc) {
        super(handle, data, inc);
        this.data = data;
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
        super(handle, (batchSize - 1) * strideSize + inc * subVecDim);
        this.data = dArray().getAsBatch(strideSize, batchSize, subVecDim);
    }


    /**
     * The element at the ith index in every subVector.
     *
     * @param i The index of the desired element.
     * @return The element at the ith index in every subVector.
     */
    public Vector getElement(int i) {
        return new Vector(getHandle(), data.subArray(i * inc), data.stride * inc);
    }
    
    /**
     * Gets the subvector at the desired index.
     * @param i The index of the desired subvector.
     * @return The subvector at the desired index.
     */
    public Vector getVector(int i){
        return getSubVector(data.stride*i/inc, getSubVecDim());
    }

    /**
     * The dimension of the subarrays.
     *
     * @return The number of elements in each sub array.
     */
    public int getSubVecDim() {
        return Math.ceilDiv(data.subArrayLength, inc);
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
    public void addMatVecMult(boolean transposeMats, MatricesStride mats, VectorsStride vecs, double timesAB, double timesThis) {
        data.multMatMatStridedBatched(getHandle(), transposeMats, true,
                transposeMats?mats.width:mats.height, getSubVecDim(), 1,
                timesAB,
                mats.getBatchArray(), mats.getColDist(),
                vecs.data, vecs.inc,
                timesThis, inc
        );
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
    public void addVecMatMult(boolean transposeMats, VectorsStride vecs, MatricesStride mats, double timesAB, double timesThis) {
        data.multMatMatStridedBatched(getHandle(), false, transposeMats,
                1, 
                transposeMats?mats.width:mats.height, 
                getSubVecDim(),
                timesAB,
                vecs.data, vecs.inc,
                mats.getBatchArray(), mats.getColDist(),
                timesThis, inc
        );
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     */
    public void setVecMatMult(VectorsStride vecs, MatricesStride mats) {
        addVecMatMult(false, vecs, mats, 1, 0);
    }

    /**
     * Multiplies a set of matrices by a set of vectors. The result will be put
     * in this.
     *
     * @param mats The matrices.
     * @param vecs The vectors.
     */
    public void setMatVecMult(MatricesStride mats, VectorsStride vecs) {
        addMatVecMult(false, mats, vecs, 1, 0);
    }

    public static void main(String[] args) {
        try(
                Handle hand = new Handle(); 
                DArray array = new DArray(hand, 1,2,  3,4,     5,6,  7,8)
                ){
            
                Vector vec = new Vector(hand, array, 2);
                VectorsStride vs = vec.subVectors(2, 2, 2, 1);
                System.out.println(vs.getVector(1));
        }
    }
    
    /**
     * Partitions this into an array of vectors so that v[i].get(j) is the ith
     * element in the jth vector.
     *
     * @return Partitions this into an array of vectors so that v[i].get(j) is
     * the ith element in the jth vector.
     */
    public Vector[] parition() {
        Vector[] parts = new Vector[getSubVecDim()];
        Arrays.setAll(parts, i -> getSubVector(i * inc, data.batchSize, data.stride));
        return parts;
    }
    
    /**
     * A contiguous subset of the subvectors in this set.
     * @param start The index of the first subvector.
     * @param length The number of subvectors.
     * @return The subset.
     */
    public VectorsStride subBatch(int start, int length){
        return new VectorsStride(getHandle(), data.subBatch(start, length), inc);
    }

}
