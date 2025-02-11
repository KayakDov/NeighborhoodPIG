package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.stream.IntStream;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.DStrideArray3d;

/**
 * This stack of matrices is stored by default as follows: A group of columns
 * stored sequentially form a layer. A group of layers form a tensor, and a
 * group of tensors form the TensorStride.
 *
 * @author Dov Neimand
 */
public class TensorOrd3Stride extends TensorOrd3StrideDim implements AutoCloseable, ColumnMajor {

    protected final DStrideArray3d data;

    /**
     * A set of order 3d tensors.
     *
     * @param handle The handle.
     * @param data The underlying data.
     */
    public TensorOrd3Stride(Handle handle, DStrideArray3d data) {
        super(handle, data.entriesPerLine(), data.linesPerLayer(), data.numLayers, data.linesPerLayer()*data.ld(), data.stride(), data.batchSize());
        this.data = data;
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
        this(handle, new DStrideArray3d(height, width, depth, batchSize));
    }

    /**
     * Copies the dimensions of this, but leaves the data empty.
     *
     * @return An empty TensorOrd3Stride with the same dimensions as this.
     */
    public TensorOrd3Stride emptyCopyDimensions() {
        return copyDimensions(data.copyDim());
    }

    /**
     * Imposes the dimensions of this onto the proffered data.
     *
     * @param array The data that will be fit into these dimensions. Make sure
     * the array is long enough to receive these dimensions.
     * @return An empty TensorOrd3Stride with the same dimensions as this.
     */
    public TensorOrd3Stride copyDimensions(DArray array) {
        return new TensorOrd3Stride(handle, new DStrideArray3d(array, height, width, depth, batchSize));
    }

    /**
     * Finds the row of the column-major index.
     *
     * @param index The column major index for which the row is desired.
     */
    public int rowFromInd(int index) {
        return ((index % strideSize) % layerDist) % colDist();
    }

    /**
     * Finds the column of the column-major index.
     *
     * @param index The column major index for which the column is desired.
     */
    public int colFromInd(int index) {
        return ((index % strideSize) % layerDist) / colDist();
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
     * Closes the underlying data.
     */
    public void close() {
        data.close();
    }

    /**
     * The underlying data behind this object.
     *
     * @return The underlying data behind this object.
     */
    public DStrideArray3d array() {
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
        return new TensorOrd3(handle, data.getSubArray(i));
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < batchSize; i++)
            sb.append(getTensor(i).toString()).append("\n");
        return super.toString() + sb.toString();
    }

    @Override
    public int colDist() {
        return data.ld();
    }

    /**
     * Finds the index of the first tensor with a NaN value. This is meant to be
     * a debugging tool.
     *
     * @return Finds the index of the first tensor with a NaN value.
     */
    public int firstTensorIndexOfNaN() {
        double[] cpuData = data.get(handle);

        int arrayIndexOfFirstNaN = IntStream.range(0, cpuData.length).filter(i -> Double.isNaN(cpuData[i]) || !Double.isFinite(cpuData[i])).findFirst().orElse(-1);

        return arrayIndexOfFirstNaN / strideSize;
    }

}
