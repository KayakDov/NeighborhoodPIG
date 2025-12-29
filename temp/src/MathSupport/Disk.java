package MathSupport;

import imageWork.Pencil;

/**
 * Represents a 3D disk defined by a center point, a normal vector (represented as Point3d), and a radius.
 * This class is designed to be similar in structure to the provided Interval class.
 *
 * This version prioritizes memory efficiency by directly assigning Point3d references
 * for center and normal, rather than creating defensive copies. This means external
 * modifications to the Point3d objects passed into the set method
 * will directly affect the internal state of the Disk object.
 *
 * Assumes the existence of:
 * - MathSupport.Point3d: A class representing a 3D point with methods like set, x(), y(), z(),
 * scale(), translate(), and `cross(Point3d a, Point3d b)` (which sets 'this' to A x B).
 * It also includes `translate(double scalar, Point3d p)`.
 * - imageWork.vectors.VectorImg.Pencil: An interface or class with a 'mark(Point3d, int)' method
 * for drawing individual points.
 *
 * IMPORTANT: This class assumes that any normal vectors provided to its set method
 * are ALREADY NORMALIZED (i.e., unit vectors). No internal normalization
 * is performed to save computation.
 * IMPORTANT: This class assumes that any radius provided to its set method
 * is POSITIVE. No internal clamping to non-negative is performed.
 */
public class Disk {
    private Point3d center;
    private Point3d normal;
    private double radius;

    /**
     * Sets the properties of the disk. This method is designed for speed and memory efficiency.
     * Direct assignment for memory efficiency.
     *
     * @param center The new center point of the disk. The reference is directly assigned.
     * @param normal The new normal vector to the disk's plane, represented as a Point3d.
     * This vector is assumed to be already normalized.
     * @param radius The new radius of the disk. Must be positive.
     * @return This Disk instance, allowing for method chaining.
     */
    public Disk set(Point3d center, Point3d normal, double radius) {
        this.center = center;
        this.normal = normal;
        this.radius = radius;
        return this;
    }

    /**
     * Returns the disk's center point.
     *
     * @return The internal Point3d object representing the center.
     */
    public Point3d getCenter() {
        return center;
    }

    /**
     * Returns the disk's normal vector.
     *
     * @return The internal Point3d object representing the normal.
     */
    public Point3d getNormal() {
        return normal;
    }

    /**
     * Returns the radius of the disk.
     * @return The radius.
     */
    public double getRadius() {
        return radius;
    }

    /**
     * Sets up two orthogonal basis vectors (u and v) that lie in the plane of the disk.
     * These vectors, along with the disk's normal, form an orthonormal basis for the disk's plane.
     * This method modifies the provided Point3d buffers in place.
     *
     * @param u A pre-allocated Point3d object to store the first basis vector. This will be modified.
     * @param v A pre-allocated Point3d object to store the second basis vector. This will be modified.
     * @param z A pre-allocated Point3d object to represent the Z-axis (0,0,1) for calculations. This will be modified.
     */
    private void setupBasisVectors(Point3d u, Point3d v, Point3d z) {
        z.set(0, 0, 1);

        if (u.set(z).translate(-1, normal).normSq() <= 1e-6) {
            u.set(1, 0, 0);
            v.set(0, 1, 0);
        } else {
            u.cross(normal, z);
            v.cross(normal, u);
        }
    }

    /**
     * Draws points along a circle segment. This is a helper method used by drawFilled and draw.
     *
     * @param pen A Pencil object from VectorImg, used to mark individual points.
     * @param loc A pre-allocated Point3d object used as a buffer for the
     * point to be drawn. This point will be modified.
     * @param u A pre-allocated Point3d object representing the first basis vector in the disk's plane.
     * @param v A pre-allocated Point3d object representing the second basis vector in the disk's plane.
     * @param r The radius of the circle to draw.
     * @param t The time (or frame number) passed to the Pencil's mark method.
     */
    private void drawCircleSegment(Pencil pen, Point3d loc, Point3d u, Point3d v, double r, int t) {
        double circumference = 2 * Math.PI * r;
        int numAngleSteps = (int) Math.max(4, Math.round(circumference * 2));
        if (numAngleSteps == 0) numAngleSteps = 1;

        for (double theta = 0; theta < numAngleSteps; theta += 2 * Math.PI / numAngleSteps) {
            loc.set(center).translate(r * Math.cos(theta), u).translate(r * Math.sin(theta), v);
            pen.mark(loc, t);
        }
    }

    /**
     * Draws points to fill the surface of the disk using concentric circles.
     * This method iterates through various radii and angles to draw points,
     * effectively filling the disk with points. All intermediate calculations
     * use provided pre-allocated buffers to avoid heap memory allocation.
     *
     * @param pen A Pencil object from VectorImg, used to mark individual points.
     * @param loc A pre-allocated Point3d object used as a buffer for the
     * point to be drawn. This point will be modified.
     * @param u A pre-allocated Point3d object to store the first basis vector
     * in the disk's plane. This point will be modified.
     * @param v A pre-allocated Point3d object to store the second basis vector
     * in the disk's plane. This point will be modified.
     * @param t The time (or frame number) passed to the Pencil's mark method.
     */
    public void drawFilled(Pencil pen, Point3d loc, Point3d u, Point3d v, int t) {
        setupBasisVectors(u, v, loc);

        int numRadiusSteps = (int) Math.max(1, Math.round(radius * 2));

        pen.mark(center, t);

        // Iterate through concentric circles from the first step outwards to the full radius.
        for (int i = 1; i <= numRadiusSteps; i++) {
            double r = (double) i * radius / numRadiusSteps; // Current radius for this circle
            drawCircleSegment(pen, loc, u, v, r, t);
        }
    }

    /**
     * Draws only the outer circumference of the disk.
     * All intermediate calculations use provided pre-allocated buffers to avoid heap memory allocation.
     *
     * @param pen A Pencil object from VectorImg, used to mark individual points.
     * @param loc A pre-allocated Point3d object used as a buffer for the
     * point to be drawn. This point will be modified.
     * @param u A pre-allocated Point3d object to store the first basis vector
     * in the disk's plane. This point will be modified.
     * @param v A pre-allocated Point3d object to store the second basis vector
     * in the disk's plane. This point will be modified.
     * @param t The time (or frame number) passed to the Pencil's mark method.
     */
    public void draw(Pencil pen, Point3d loc, Point3d u, Point3d v, int t) {
        setupBasisVectors(u, v, loc);

        double r = radius;
        drawCircleSegment(pen, loc, u, v, r, t);
    }

    /**
     * Returns a string representation of the Disk object.
     * @return A string detailing the disk's center, normal, and radius.
     */
    @Override
    public String toString() {
        return "Disk [Center: " + center.toString() +
               ", Normal: " + normal.toString() +
               ", Radius: " + String.format("%.2f", radius) + "]";
    }
}
