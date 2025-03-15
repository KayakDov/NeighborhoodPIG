package fijiPlugin;

/**
 * The dimensions of a neighborhood for the purpose of computing the structure
 * tensor of a pixel. Each dimension is the distance from the center to the
 * nearest edge of the cube.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodDim {

    public final int xyR, zR;
    public final double layerRes;

    /**
     * @param xy The distance to the nearest side of the xy plane neighborhood
     * square.
     * @param z The distance to the nearest xy parallel cube surface.
     * @param distBetweenAdjLayer The distance between adjacent layers as a multiple of the distance between xy pixels.
     */
    public NeighborhoodDim(int xy, int z, double distBetweenAdjLayer) {
        this.xyR = xy;
        this.zR = z;
        this.layerRes = distBetweenAdjLayer;
    }

}
