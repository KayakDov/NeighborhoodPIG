package fijiPlugin;

import java.util.Optional;

/**
 * The dimensions of a neighborhood for the purpose of computing the structure
 * tensor of a pixel. Each dimension is the distance from the center to the
 * nearest edge of the cube.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodDim {

    public final int xyR;
    public Optional<Integer> zR;
    public final Optional<Double> layerRes;

    /**
     * @param xy The distance to the nearest side of the xy plane neighborhood
     * square.
     * @param z The distance to the nearest xy parallel cube surface.
     * @param distBetweenAdjLayer The distance between adjacent layers as a
     * multiple of the distance between xy pixels.
     */
    public NeighborhoodDim(int xy, Optional<Integer> z, Optional<Double> distBetweenAdjLayer) {
        this.xyR = xy;
        this.zR = z;
        this.layerRes = distBetweenAdjLayer;
    }

    /**
     * Checks if the values stored here are valid.
     *
     * @return True if all values are positive.
     */
    public boolean valid() {
        return xyR >= 0 && zR.orElse(1) >= 0 && layerRes.orElse(1.0) > 0;
    }

    /**
     * Returns a string representation of the NeighborhoodDim object.
     *
     * @return A string containing the values of the NeighborhoodDim's fields.
     */
    @Override
    public String toString() {
        return "NeighborhoodDim{"
                + "rX=" + xyR
                + ", rY=" + xyR
                + (zR.isPresent()? ", rZ=" + zR.get():"")                
                + (layerRes.isPresent()?", layerSpacing=" + layerRes:"")
                + '}';
    }
}
