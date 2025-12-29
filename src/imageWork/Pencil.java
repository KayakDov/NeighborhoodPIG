package imageWork;

import MathSupport.Point3d;
import java.awt.Color;

/**
     * An object to facilitate drawing with IJ at the proffered point.
     */
    public interface Pencil {

        /**
         * Sets the color of the vector.
         *
         * @param color The new color of the vector.
         */
        public void setColor(Color color);

        /**
         * Sets the color of the vector.
         *
         * @param color The new color of the vector.
         */
        public void setColor(int[] color);

        /**
         * Draws the preset color to the desired place and time.
         *
         * @param p The location to draw.
         * @param t The frame to draw on.
         */
        public void mark(Point3d p, int t);
    }