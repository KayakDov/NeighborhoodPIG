package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;
import jcuda.jcublas.JCublas2;

/**
 *
 * @author E. Dov Neimand
 */
public interface DLineArray extends DArray {

    
    /**
     * Fills a matrix with a scalar value directly on the GPU using a CUDA
     * kernel.
     *
     * This function sets all elements of the matrix A to the given scalar
     * value. The matrix A is stored in column-major order, and the leading
     * dimension of A is specified by lda.
     *
     * In contrast to the method that doesn't use a handle, this one
     *
     * @param handle A handle.
     * @param fill the scalar value to set all elements of A
     * @param inc The increment with which the method iterates over the array.
     * @return this;
     */
    public default DLineArray fill(Handle handle, double fill, int inc) {

        fill(handle, fill, inc, entriesPerLine(), ld());
        return this;
    }

    /**
     * The number of entries that could be put on a line if they were to use the
     * entire space of the pitch.
     *
     * @return The number of entries that could be put on a line if they were to
     * use the entire space of the pitch.
     */
    public default int ld() {
        return bytesPerLine() / Sizeof.DOUBLE;
    }

    /**
     *
     * @param lineIndex
     * @return
     */
    public default DArray1d getLine(int lineIndex) {
        return new DArray1d(this, ld() * lineIndex, entriesPerLine());
    }

    /**
     * {@inheritDoc }
     */
    public default DArray1d as1d() {
        return new DArray1d(this, 0, 1);
    }

    /**
     * A 2 dimensional representation of this array. If this array is already
     * 2d, then this array is returned. If it is 3d then each layer precedes the
     * previouse layers.
     *
     * @return A 2 dimensional representation of this array.
     */
    public default DArray2d as2d() {
        return new DArray2d(this, entriesPerLine());
    }
    
    /**
     * A 3d representation of this array.
     * @param linesPerLayer
     * @return 
     */
    public default DArray3d as3d(int linesPerLayer){
        return new DArray3d(this, entriesPerLine(), linesPerLayer);
    }
}
