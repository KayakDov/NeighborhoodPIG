package imageWork;

import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;

/**
 * Inner class to manage vector data.
 */
public class VecManager extends Dimensions{

    private final float[] vecs;

    /**
     * Constructs a new VecManager with the specified size.
     *
     * @param d The dimensions (number and organization of 3d vectors) in this manager
     */
    public VecManager(Dimensions d) {
        super(d);
        this.vecs = new float[d.tensorSize()*3];
    }

    /**
     * Sets the vector data from the given {@link FStrideArray3d} at the
     * specified grid index.
     *
     * @param gpuStrideArray The {@link FStrideArray3d} containing the
     * vector data.
     * @param gridIndex The index of the grid to retrieve data from.
     * @param handle
     * @return this.
     */
    public VecManager setFrom(FStrideArray3d gpuStrideArray, int gridIndex, Handle handle) {
        gpuStrideArray.getGrid(gridIndex).get(handle, vecs);
        return this;
    }

    /**
     * Calculates the index of a vector in the vector data array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param layer The layer index.
     * @return The index of the vector.
     */
    public int vecIndex(int row, int col, int layer) {
        return (layer * layerSize() + col * height + row) * 3;
    }

    /**
     * Retrieves the vector at the specified row, column, and layer and
     * copies it to the provided array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param layer The layer index.
     * @param vec The array to store the retrieved vector.
     */
    public void get(int row, int col, int layer, float[] vec) {
        System.arraycopy(vecs, vecIndex(row, col, layer), vec, 0, 3);
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
    public float[] get(int row, int col, int layer) {
        float[] vec = new float[3];
        get(row, col, layer, vec);
        return vec;
    }

    /**
     * Retrieves the vector at the specified row, column, and layer and
     * copies it to the provided array.
     *
     * @param row The row index.
     * @param col The column index.
     * @param layer The layer index.
     * @param p The point the values are to be assigned to.
     * @param scale Multiply the vector before it is rounded to ints.
     */
    public void get(int row, int col, int layer, Point3d p, double scale){
        int ind = vecIndex(row, col, layer);
        p.set(scale * vecs[ind], scale * vecs[ind + 1], scale * vecs[ind + 2]);
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        Point3d p = new Point3d();
        for (int row = 0; row < height; row++)
            for (int col = 0; col < height; col++)
                for (int layer = 0; layer < depth; layer++) {
                    get(row, col, layer, p, 100);
                    sb.append("\n").append(p);
                }
        return sb.toString();
    }
    
}
