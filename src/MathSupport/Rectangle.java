package MathSupport;

/**
 *
 * @author E. Dov Neimand
 */
public class Rectangle extends Cube{
    /**
     * Constructs a Cube with the specified minimum and maximum coordinates.
     *
     * @param minX the minimum x-coordinate (inclusive)
     * @param maxX the maximum x-coordinate (exclusive)
     * @param minY the minimum y-coordinate (inclusive)
     * @param maxY the maximum y-coordinate (exclusive)
     */
    public Rectangle(int minX, int maxX, int minY, int maxY) {
        super(minX, maxX, minY, maxY, 0, 1);
    }

    /**
     * Constructs a Cube with its origin at (0, 0, 0) and the specified dimensions.
     *
     * @param width  the width of the cube (extent along the x-axis)
     * @param height the height of the cube (extent along the y-axis)
     */
    public Rectangle(int width, int height) {
        this(0, width, 0, height);
    }
}
