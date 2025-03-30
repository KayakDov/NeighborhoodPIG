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
    public final float layerRes;

    /**
     * @param xy The distance to the nearest side of the xy plane neighborhood
     * square.
     * @param z The distance to the nearest xy parallel cube surface.
     * @param distBetweenAdjLayer The distance between adjacent layers as a multiple of the distance between xy pixels.
     */
    public NeighborhoodDim(int xy, int z, float distBetweenAdjLayer) {
        this.xyR = xy;
        this.zR = z;
        this.layerRes = distBetweenAdjLayer;
    }

    /**
     * Checks if the values stored here are valid.
     * @return True if all values are positive.
     */
    public boolean valid(){
        return xyR > 0 && zR > 0 && layerRes > 0;
    }
}
