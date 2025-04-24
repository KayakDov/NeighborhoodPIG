package MathSupport;

/**
 * Represents an axis-aligned cube (or rectangular prism) in 3-dimensional integer space.
 * <p>
 * The cube is defined by its minimum and maximum coordinates along the X, Y, and Z axes.
 * It contains all points (x, y, z) satisfying:
 * <pre>
 *   min.getX() ≤ x < max.getX(),
 *   min.getY() ≤ y < max.getY(),
 *   min.getZ() ≤ z < max.getZ().
 * </pre>
 * </p>
 * 
 * @author E. Dov Neimand
 */
public class Cube {

    private Point3d max, min;

    /**
     * Constructs a cube.
     * @param minX the x value for a corner.
     * @param maxX the x value for the other corner.
     * @param minY a y value for a corner.
     * @param maxY the y value for the other corner.
     * @param minZ a z value for a corner.
     * @param maxZ The z value for the other corner.
     */
    public Cube(int minX, int maxX, int minY, int maxY, int minZ, int maxZ) {
        min = new Point3d(minX, minY, minZ);
        max = new Point3d(maxX, maxY, maxZ);
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
    public int width() {
        return max.x() - min.x();
    }
    
    /**
     * Returns the height of the cube (the extent along the y-axis).
     *
     * @return the height of the cube
     */
    public int height() {
        return max.y() - min.y();
    }
    
    /**
     * Returns the depth of the cube (the extent along the z-axis).
     *
     * @return the depth of the cube
     */
    public int depth() {
        return max.z() - min.z();
    }
    
    /**
     * Checks whether the cube contains the given point.
     * <p>
     * A point (x, y, z) is considered inside the cube if:
     * <pre>
     *   min.getX() ≤ x < max.getX(),
     *   min.getY() ≤ y < max.getY(),
     *   min.getZ() ≤ z < max.getZ().
     * </pre>
     * </p>
     *
     * @param x the x-coordinate of the point to test
     * @param y the y-coordinate of the point to test
     * @param z the z-coordinate of the point to test
     * @return {@code true} if the point is within the cube, {@code false} otherwise
     */
    public boolean contains(int x, int y, int z) {
        return x >= min.x() && x < max.x() &&
               y >= min.y() && y < max.y() &&
               z >= min.z() && z < max.z();
    }

    /**
     * Returns a string representation of the cube.
     *
     * @return a string describing the cube's bounds
     */
    @Override
    public String toString() {
        return "Cube [min=(" + min.x() + ", " + min.y() + ", " + min.z() + "), "
                + "max=(" + max.x() + ", " + max.y() + ", " + max.z() + ")]";
    }
    
    /**
     * Translates this cube.
     * @param xShift
     * @param yShift
     * @param zShift
     * @return 
     */
    public void translate(int xShift, int yShift, int zShift){
        min.translate(xShift, yShift, zShift);
        max.translate(xShift, yShift, zShift);
    }
    /**
     * Creates a new scew that is this cube scaled.
     * @param t The scalar.
     * @return  A new cube.
     */
    public void scale(int t){
        min.scale(t);
        max.scale(t);
    }
    
    
    /**
     * The distance between the min corner and the max corner squared.
     * @return The distance between the min corner and the max corner squared.
     */
    public int distSquared(){
        return min.distSquared(max);
    }
    
    public void setMin(int x, int y, int z){
        min.set(x, y, z);
    }
    
    public void setMax(int x, int y, int z){
        max.set(x, y, z);
    }

    public int getMaxX() {
        return max.x();
    }

    public int getMaxY() {
        return max.y();
    }

    public int getMaxZ() {
        return max.z();
    }

    public int getMinX() {
        return min.x();
    }

    public int getMinY() {
        return min.y();
    }

    public int getMinZ() {
        return min.z();
    }

    public Cube() {
    }
    
    
}
