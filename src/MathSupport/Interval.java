package MathSupport;

import imageWork.VectorImg;

/**
 *
 * @author E. Dov Neimand
 */
public class Interval {

    private Point3d a, b;

    /**
     * Creates the line.
     *
     * @param a One end point.
     * @param b The other end point.
     */
    public Interval(Point3d a, Point3d b) {
        this.a = a;
        this.b = b;
    }

    public Interval() {
        a = new Point3d();
        b = new Point3d();
    }

    public void setA(int x, int y, int z) {
        a.set(x, y, z);
    }

    public void setB(int x, int y, int z) {
        b.set(x, y, z);
    }

    /**
     * Sets a proffered point to be along this line. setMe = (1 - t) a + t b
     *
     * @param t Where between a and b should the point be set to,.
     * @param setMe
     */
    public void at(double t, Point3d setMe) {
        setMe.set(a).scale(1 - t).translate(b.getX() * t, b.getY() * t, b.getZ() * t);
    }

    /**
     * Draws points along the line.
     *
     * @param pen A method that can draw a point.
     * @param vector Pre allocated memory so this method can run faster if
     * called many times in a row.
     * @param delta Pre allocated memory so this method can run faster if called
     * many times in a row.
     * @param t The time (frame) to draw at.
     */
    public void draw(VectorImg.Pencil pen, Point3d vector, Point3d delta, int t) {
        double dist = length();
               
        pen.mark(vector.set(a), t);

        delta.set(b).translate(-1, a).scale(1 / dist);
        
        for (int x = 0; x < dist; x++)
            pen.mark(vector.translate(delta), t);
    }

    public Point3d getA() {
        return a;
    }

    public Point3d getB() {
        return b;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String toString() {
        return a.toString() + " : " + b.toString() + " with length " + length();
    }

    /**
     * The length of the line.
     *
     * @return The length of the line.
     */
    public double length() {
        return a.difference(b).norm();
    }

}
