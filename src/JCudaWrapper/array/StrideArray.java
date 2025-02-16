package JCudaWrapper.array;

/**
 *
 * @author dov
 */
public interface StrideArray extends Array {

    /**
     * The stride size.
     * @return The stride size.
     */
    public int strideLines();
    
    /**
     * The batch Size. The number of strides taken.
     * @return The batch Size;The number of strides taken.
     */
    public int batchSize();
    
    /**
     * The length of each sub array.
     * @return The length of each sub array.
     */
    public int subArraySize();

    /**
     * The sub array the given stride.
     * @param arrayIndex The stride of the desired subarray.
     * @return The sub array the given stride.
     */
    public Array getSubArray(int arrayIndex);
    
    
    /**
     * The length of the array.
     *
     * @param batchSize The number of elements in the batch.
     * @param strideSize The distance between the first elements of each batch.
     * @param subArrayLength The length of each subArray.
     * @return The minimum length to hold a batch described by these paramters.
     */
    public static int totalDataLength(int strideSize, int subArrayLength, int batchSize) {
        return strideSize * (batchSize - 1) + subArrayLength;
    }
    
}
