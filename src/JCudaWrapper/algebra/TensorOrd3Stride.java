package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.resourceManagement.Handle;

/**
 * This stack of matrices is stored by default as follows: A group of columns
 * stored sequentially form a layer. A group of layers form a tensor, and a
 * group of tensors form the TensorStride.
 *
 * @author Dov Neimand
 */
public class TensorOrd3Stride extends TensorOrd3dStrideDim implements AutoCloseable, ColumnMajor {
        
    protected final DStrideArray data;    

    /**
     * A set of order 3 tensors.
     *
     * @param handle The handle.
     * @param height The height of the tensors.
     * @param width The width of the tensors.
     * @param depth The depth of the tensors.
     * @param colDist The distance between the first elements of the columns of
     * the tensors.
     * @param layerDist The distance between the first elements of the layers of
     * the tensors.
     * @param strideSize The distance between the first elements of the tensors.
     * @param batchSize The number of tensors.
     * @param data The underlying data.
     */
    public TensorOrd3Stride(Handle handle, int height, int width, int depth, int colDist, int layerDist, int strideSize, int batchSize, DArray data) {
        super(handle, height, width, depth, colDist, layerDist, strideSize, batchSize);
        this.data = data.getAsBatch(strideSize, layerDist * (depth - 1) + colDist * (width - 1) + height, batchSize);
    }

    /**
     * A set of order 3 tensors.
     *
     * @param handle The handle.
     * @param height The height of the tensors.
     * @param width The width of the tensors.
     * @param depth The depth of the tensors.
     * @param batchSize The number of tensors.
     * @param data The underlying data.
     */
    public TensorOrd3Stride(Handle handle, int height, int width, int depth, int batchSize, DArray data) {
        this(handle, height, width, depth, height, height * width, height * width * depth, batchSize, data);
    }

    /**
     * A set of order 3 tensors.
     *
     * @param handle The handle.
     * @param height The height of the tensors.
     * @param width The width of the tensors.
     * @param depth The depth of the tensors.
     * @param batchSize The number of tensors.
     */
    public TensorOrd3Stride(Handle handle, int height, int width, int depth, int batchSize) {
        this(handle, height, width, depth, height,
                height * width, height * width * depth, batchSize,
                DArray.empty(height * width * depth * batchSize)
                        .getAsBatch(height * width * depth, batchSize)
        );
    }

    /**
     * Copies the dimensions of this, but leaves the data empty.
     *
     * @return An empty TensorOrd3Stride with the same dimensions as this.
     */
    public TensorOrd3Stride emptyCopyDimensions() {
        return new TensorOrd3Stride(handle, height, width, depth, batchSize);
    }

    /**
     * Finds the row of the column-major index.
     *
     * @param index The column major index for which the row is desired.
     */
    public int rowFromInd(int index) {
        return ((index % strideSize) % layerDist) % colDist;
    }

    /**
     * Finds the column of the column-major index.
     *
     * @param index The column major index for which the column is desired.
     */
    public int colFromInd(int index) {
        return ((index % strideSize) % layerDist) / colDist;
    }

    /**
     * Finds the layer of the column-major index.
     *
     * @param index The column major index for which the layer is desired.
     */
    public int layerFromInd(int index) {
        return (index % strideSize) / layerDist;
    }

    /**
     * Finds the tensor of the column-major index.
     *
     * @param index The column major index for which the tensor is desired.
     */
    public int tensorFromInd(int index) {
        return index / strideSize;
    }

    /**
     * A sequence of sets of consecutive slices orthogonal to depth.
     *
     * @param firstLayerInd The index of the slices in each tensor.
     * @param numConsecutiveSlices The number of slices taken from each matrix.
     * @return A sequence of slices at the desired depth.
     */
    public TensorOrd3Stride layerSequence(int firstLayerInd, int numConsecutiveSlices) {
        return new TensorOrd3Stride(handle,
                height, width, numConsecutiveSlices,
                colDist, layerDist, strideSize,
                batchSize,
                data.subArray(layerDist * firstLayerInd)
        );
    }

    /**
     * A sequence of sets of consecutive slices orthogonal to a row.
     *
     * @param firstColInd The index of the slices in each tensor.
     * @param numConsecutiveSlices The number of slices in each set.
     * @return A sequence of slices at the desired column.
     */
    public TensorOrd3Stride depthXColSequence(int firstColInd, int numConsecutiveSlices) {
        return new TensorOrd3Stride(handle,
                height, numConsecutiveSlices, depth,
                colDist, layerDist, strideSize,
                batchSize,
                data.subArray(colDist * firstColInd)
        );
    }

    /**
     * A sequence of sets of sets of consecutive slices orthogonal to a column.
     *
     * @param firstRowInd The index of the slices in each tensor.
     * @param numConsecutiveSlices The number of slices in each set.
     * @return A sequence of slices at the desired row.
     */
    public TensorOrd3Stride rowXDepthSequence(int firstRowInd, int numConsecutiveSlices) {
        return new TensorOrd3Stride(handle,
                numConsecutiveSlices, width, depth,
                colDist,
                layerDist,
                strideSize,
                batchSize,
                data.subArray(firstRowInd)
        );
    }

    /**
     * Closes the underlying data.
     */
    @Override
    public void close() {
        data.close();
    }

    /**
     * The underlying data behind this object.
     *
     * @return The underlying data behind this object.
     */
    public DStrideArray dArray() {
        return data;
    }

    /**
     * The number of elements.
     *
     * @return The number of elements.
     */
    public int size() {
        return height * width * depth * batchSize;
    }

    /**
     * Gets the tensor at the given index.
     *
     * @param i The index of the desired tensor.
     * @return The tensor at the given index.
     */
    public TensorOrd3 getTensor(int i) {
        return new TensorOrd3(handle,
                data.subArray(
                        i * strideSize,
                        i * strideSize + layerDist * (depth - 1) + colDist * (width - 1) + height),
                height, width, depth, colDist, layerDist
        );
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < batchSize; i++)
            sb.append(getTensor(i).toString()).append("\n");
        return sb.toString();
    }

    @Override
    public int getColDist() {
        return colDist;
    }

}
