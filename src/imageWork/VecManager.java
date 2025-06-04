package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;

/**
 * class to manage vector data.
 */
public class VecManager {

    private final float[] vecs;
    private Dimensions dim;

    /**
     * Constructs a new VecManager with the specified size.
     *
     * @param d The dimensions (number and organization of 3d vectors) in this manager
     */
    public VecManager(Dimensions d) {
        this.dim = d;
        this.vecs = new float[d.tensorSize()*dim.num()];
    }

    /**
     * Sets the vector data from the given {@link FStrideArray3d} at the
     * specified grid index.
     *
     * @param gpuVecs The {@link FStrideArray3d} containing the
     * vector data.
     * @param gridIndex The index of the grid to retrieve data from.
     * @param layerIndex The index of the layer.
     * @param handle
     * @return this.
     */
    public VecManager setFrom(PArray2dToF2d gpuVecs, int gridIndex, int layerIndex, Handle handle) {
        gpuVecs.get(layerIndex, gridIndex).getVal(handle).get(handle, vecs);
        return this;
    }

    /**
     * Calculates the index of a vector in the vector data array.
     *
     * @param row The row index.
     * @param col The column index.
     * @return The index of the vector.
     */
    public int vecIndex(int row, int col) {
        return (col * dim.height + row) * dim.num();
    }

    /**
     * Retrieves the vector at the specified row, column, and layer and
     * copies it to the provided array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param vec The array to store the retrieved vector.
     */
    public void get(int row, int col, double[] vec) {
        System.arraycopy(vecs, vecIndex(row, col), vec, 0, dim.num());
    }
    
    
    /**
     * Retrieves the vector at the specified row, column, and layer and
     * copies it to the provided array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param layer The layer index.
     * @return the vector.
     */
    public double[] get(int row, int col, int layer) {
        double[] vec = new double[dim.num()];
        get(row, col, vec);
        return vec;
    }

    /**
     * Retrieves the vector at the specified row, column, and layer and
     * copies it to the provided array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param p The point the values are to be assigned to.
     * @param scale Multiply the vector before it is rounded to ints.
     */
    public void get(int row, int col,  Point3d p, double scale){
        int ind = vecIndex(row, col);
        p.set(scale * vecs[ind], scale * vecs[ind + 1], dim.hasDepth() ? scale * vecs[ind + 2]:0);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Point3d p = new Point3d();
        for (int row = 0; row < dim.height; row++)
            for (int col = 0; col < dim.height; col++)
                for (int layer = 0; layer < dim.depth; layer++) {
                    get(row, col,p, 100);
                    sb.append("\n").append(p);
                }
        return sb.toString();
    }
    
}
