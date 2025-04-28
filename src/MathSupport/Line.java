package MathSupport;

import java.util.function.Consumer;

/**
 *
 * @author E. Dov Neimand
 */
public class Line {

    private Point3d a, b;

    /**
     * Creates the line.
     *
     * @param a One end point.
     * @param b The other end point.
     */
    public Line(Point3d a, Point3d b) {
        this.a = a;
        this.b = b;
    }

    public Line() {
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
     * @param cursor Pre allocated memory so this method can run faster if
     * called many times in a row.
     * @param delta Pre allocated memory so this method can run faster if called
     * many times in a row.
     */
    public void draw(Consumer<Point3d> pen, Point3d cursor, Point3d delta) {
        double dist = length();

        
        
        pen.accept(cursor.set(a));

        delta.set(b).translate(-1, a).scale(1 / dist);

//        System.out.println("MathSupport.Line.draw() length = " + dist + " delta = " + delta);
        
        for (int i = 0; i < dist; i++)
            pen.accept(cursor.translate(delta));
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
