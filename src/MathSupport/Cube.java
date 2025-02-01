package MathSupport;

/**
 * Represents an axis-aligned cube (or rectangular prism) in 3-dimensional integer space.
 * <p>
 * The cube is defined by its minimum and maximum coordinates along the X, Y, and Z axes.
 * It contains all points (x, y, z) satisfying:
 * <pre>
 *   minX ≤ x < maxX,
 *   minY ≤ y < maxY,
 *   minZ ≤ z < maxZ.
 * </pre>
 * </p>
 * 
 * @author E. Dov Neimand
 */
public class Cube {

    /**
     * The minimum x-coordinate (inclusive).
     */
    public final int minX;
    
    /**
     * The maximum x-coordinate (exclusive).
     */
    public final int maxX;
    
    /**
     * The minimum y-coordinate (inclusive).
     */
    public final int minY;
    
    /**
     * The maximum y-coordinate (exclusive).
     */
    public final int maxY;
    
    /**
     * The minimum z-coordinate (inclusive).
     */
    public final int minZ;
    
    /**
     * The maximum z-coordinate (exclusive).
     */
    public final int maxZ;

    /**
     * Constructs a Cube with the specified minimum and maximum coordinates.
     *
     * @param minX the minimum x-coordinate (inclusive)
     * @param maxX the maximum x-coordinate (exclusive)
     * @param minY the minimum y-coordinate (inclusive)
     * @param maxY the maximum y-coordinate (exclusive)
     * @param minZ the minimum z-coordinate (inclusive)
     * @param maxZ the maximum z-coordinate (exclusive)
     */
    public Cube(int minX, int maxX, int minY, int maxY, int minZ, int maxZ) {
        this.minX = minX;
        this.maxX = maxX;
        this.minY = minY;
        this.maxY = maxY;
        this.minZ = minZ;
        this.maxZ = maxZ;
    }

    /**
     * Constructs a Cube with its origin at (0, 0, 0) and the specified dimensions.
     *
     * @param width  the width of the cube (extent along the x-axis)
     * @param height the height of the cube (extent along the y-axis)
     * @param depth  the depth of the cube (extent along the z-axis)
     */
    public Cube(int width, int height, int depth) {
        this(0, width, 0, height, 0, depth);
    }
    
    /**
     * Returns the width of the cube (the extent along the x-axis).
     *
     * @return the width of the cube
     */
    public int getWidth() {
        return maxX - minX;
    }
    
    /**
     * Returns the height of the cube (the extent along the y-axis).
     *
     * @return the height of the cube
     */
    public int getHeight() {
        return maxY - minY;
    }
    
    /**
     * Returns the depth of the cube (the extent along the z-axis).
     *
     * @return the depth of the cube
     */
    public int getDepth() {
        return maxZ - minZ;
    }
    
    /**
     * Checks whether the cube contains the given point.
     * <p>
     * A point (x, y, z) is considered inside the cube if:
     * <pre>
     *   minX ≤ x < maxX,
     *   minY ≤ y < maxY,
     *   minZ ≤ z < maxZ.
     * </pre>
     * </p>
     *
     * @param x the x-coordinate of the point to test
     * @param y the y-coordinate of the point to test
     * @param z the z-coordinate of the point to test
     * @return {@code true} if the point is within the cube, {@code false} otherwise
     */
    public boolean contains(int x, int y, int z) {
        return x >= minX && x < maxX &&
               y >= minY && y < maxY &&
               z >= minZ && z < maxZ;
    }

    /**
     * Returns a string representation of the cube.
     *
     * @return a string describing the cube's bounds
     */
    @Override
    public String toString() {
        return "Cube [min=(" + minX + ", " + minY + ", " + minZ + "), "
                + "max=(" + maxX + ", " + maxY + ", " + maxZ + ")]";
    }
    
    /**
     * Creates a new cube that is a translation of this cube.
     * @param xShift
     * @param yShift
     * @param Shift
     * @return 
     */
    public Cube translate(int xShift, int yShift, int Shift){
        return new Cube(minX + xShift, maxX + xShift, minY + yShift, maxY + yShift, minZ + Shift, maxZ + Shift);
    }
}
