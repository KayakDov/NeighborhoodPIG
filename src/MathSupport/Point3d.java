package MathSupport;

/**
 * Represents a point in 3-dimensional integer space.
 *
 * @author E. Dov Neimand
 */
public class Point3d {

    private int x, y, z;

    /**
     * Constructs a new {@code Point3d} with the specified coordinates.
     *
     * @param x The x-coordinate of the point.
     * @param y The y-coordinate of the point.
     * @param z The z-coordinate of the point.
     */
    public Point3d(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /**
     * copies the values from another point into this one.
     *
     * @param copyMe The point to be copied.
     */
    public Point3d(Point3d copyMe) {
        x = copyMe.x;
        y = copyMe.y;
        z = copyMe.z;
    }

    /**
     * Constructs a new {@code Point3d} with default coordinates (0, 0, 0).
     */
    public Point3d() {
    }

    /**
     * Sets the coordinates of this {@code Point3d}.
     *
     * @param x The new x-coordinate.
     * @param y The new y-coordinate.
     * @param z The new z-coordinate.
     * @return this
     */
    public Point3d set(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }

    /**
     * Sets the coordinates of this {@code Point3d}.
     *
     * @param x The new x-coordinate.
     * @param y The new y-coordinate.
     * @param z The new z-coordinate.
     * @return this
     */
    public Point3d set(double x, double y, double z) {
        return set((int) Math.round(x), (int) Math.round(y), (int) Math.round(z));
    }

    /**
     * Copies the values of p into here.
     *
     * @param p The point to be copied.
     * @return this
     */
    public Point3d set(Point3d p) {
        this.x = p.x;
        this.y = p.y;
        this.z = p.z;
        return this;
    }

    /**
     * Calculates the square of an integer value.
     *
     * @param x The integer to be squared.
     * @return The square of the input integer (x*x).
     */
    private int sq(int x) {
        return x * x;
    }

    /**
     * Calculates the squared Euclidean distance between this {@code Point3d}
     * and another {@code Point3d}. Calculating the squared distance is often
     * more efficient than the actual distance as it avoids the square root
     * operation.
     *
     * @param p The other {@code Point3d} to calculate the distance to.
     * @return The squared Euclidean distance between the two points.
     */
    public int distSquared(Point3d p) {
        return sq(p.x - x) + sq(p.y - y) + sq(p.z - z);
    }

    /**
     * Gets the x-coordinate of this {@code Point3d}.
     *
     * @return The x-coordinate.
     */
    public int x() {
        return x;
    }

    /**
     * Gets the y-coordinate of this {@code Point3d}.
     *
     * @return The y-coordinate.
     */
    public int y() {
        return y;
    }

    /**
     * Gets the z-coordinate of this {@code Point3d}.
     *
     * @return The z-coordinate.
     */
    public int z() {
        return z;
    }

    /**
     * Translates this {@code Point3d} by the specified amounts in the x, y, and
     * z directions.
     *
     * @param x The amount to translate in the x-direction.
     * @param y The amount to translate in the y-direction.
     * @param z The amount to translate in the z-direction.
     * @return this
     */
    public Point3d translate(int x, int y, int z) {
        this.x += x;
        this.y += y;
        this.z += z;
        return this;
    }

    /**
     * Translates this {@code Point3d} by the specified amounts in the x, y, and
     * z directions.
     *
     * @param x The amount to translate in the x-direction.
     * @param y The amount to translate in the y-direction.
     * @param z The amount to translate in the z-direction.
     * @return this
     */
    public Point3d translate(double x, double y, double z) {
        return translate((int) Math.round(x), (int) Math.round(y), (int) Math.round(z));
    }

    /**
     * Adds anther point to this point, changing this point.
     *
     * @param p The point to be added to this point.
     * @return this.
     */
    public Point3d translate(Point3d p) {
        return translate(1, p);
    }

    /**
     * Adds anther point to this point, changing this point.
     *
     * @param scalar multiply p by this before translating.
     * @param p The point to be added to this point.
     * @return this.
     */
    public Point3d translate(float scalar, Point3d p) {
        return translate(scalar*p.x, scalar*p.y, scalar*p.z);
    }

    /**
     * Scales the coordinates of this {@code Point3d} by the given factor.Note
     * that the coordinates are integer values, so the scaling will result in
     * integer truncation.
     *
     * @param s The scaling factor.
     * @return this
     */
    public Point3d scale(double s) {
        x *= s;
        y *= s;
        z *= s;
        return this;
    }

    @Override
    public String toString() {
        return "(" + x + ", " + y + ", " + z + ")";
    }

    /**
     * Checks if all the values are non-negative.
     *
     * @return True if x, y, and z are all non negative. Otherwise, false.
     */
    public boolean firstQuadrant() {
        return x >= 0 && y >= 0 && z >= 0;
    }

    /**
     * Without changing this point, creates a new point that is the sum of this
     * point and the other point.
     *
     * @param other The point added to this one to create a new point.
     * @return The sum of this point and the other point.
     */
    public Point3d sum(Point3d other) {
        return new Point3d(other).translate(this);
    }

    /**
     * Without changing this point, creates a new point that is the difference
     * of this point and the other point.
     *
     * @param other The point subtracted from this one to create a new point.
     * @return The difference of this point and the other point.
     */
    public Point3d difference(Point3d other) {
        return new Point3d(other).scale(-1).sum(this);
    }
}
