package main;

import ij.ImagePlus;
import ij.process.ByteProcessor;
import ij.process.ImageProcessor;
import java.io.File;
import java.util.Arrays;
import java.util.Random;
import ij.io.FileSaver;
import ij.io.Opener;

/**
 * This file generates a JPEG to use for debugging purposes using the ImageJ
 * library.
 *
 * @author dov
 */
public class GenDebugFile {

    public static int[][] blackWithWhiteBorderY(int height, int width, int borderThickness) {

        int[][] black = new int[height][width];

        for (int i = 0; i < width; i++) {
            for (int j = 0; j < borderThickness; j++) {
                black[j][i] = 255;
                black[height - 1 - j][i] = 255;
            }
        }
        return black;
    }

    public static int[][] blackWithWhiteBorderX(int height, int width) {

        int[][] black = new int[height][width];
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                black[i][j] = 0;

        for (int i = 0; i < height; i++) {
            black[i][0] = 255; // left
            black[i][width - 1] = 255; // right
        }
        return black;
    }

    public static int[][] fadeToCorner(int depthDif) {
        int[][] fadeToCorner = new int[9][9];
        for (int i = 0; i < 9; i++)
            for (int j = 0; j < 9; j++)
                fadeToCorner[i][j] = (int) (255 * Math.sqrt((9 - i) * (9 - i) + (9 - j) * (9 - j) + depthDif * depthDif) / 13.0);
        return fadeToCorner;
    }

    public static int[][] uniform(int height, int width, int color) {
        int[][] uniform = new int[height][width];
        for (int i = 0; i < width; i++)
            for (int j = 0; j < height; j++)
                uniform[i][j] = color;
        return uniform;
    }

    public static int[][][] cylinder(int depth, int width, int height, int r) {
        int[][][] env = new int[depth][width][height];
        int centZ = depth / 2, centX = width / 2, centY = height / 2;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int z = 0; z < depth; z++)
                    if ((z - centZ) * (z - centZ) + (y - centY) * (y - centY) <= r * r)
                        env[z][x][y] = 255;
        return env;

    }

    /**
     * Creates a 3D integer array representing a cylinder tilted by specified
     * zenith and azimuth angles. The cylinder is centered within the 3D volume.
     *
     * @param depth The depth (size along the Z-axis, first dimension of the
     * array).
     * @param width The width (size along the X-axis, second dimension of the
     * array).
     * @param height The height (size along the Y-axis, third dimension of the
     * array).
     * @param r The radius of the cylinder.
     * @param phi The zenith angle (polar angle) in radians, measured from the
     * positive Z-axis. Ranges from 0 to Math.PI.
     * @param theta The azimuth angle in radians, measured from the positive
     * X-axis in the XY-plane. Ranges from 0 to 2 * Math.PI.
     * @return A 3D integer array where 255 indicates a point inside the
     * cylinder, and 0 otherwise. The array is indexed as
     * env[z_index][x_index][y_index].
     */
    public static int[][][] tiltedCylinder(int depth, int width, int height, int r, double phi, double theta) {
        // Initialize the 3D environment array with zeros.
        int[][][] env = new int[depth][width][height];

        // Calculate the center coordinates of the 3D volume.
        // Using double for precision in calculations.
        double centZ = depth / 2.0;
        double centX = width / 2.0;
        double centY = height / 2.0;

        // Calculate the components of the cylinder's axis direction vector (Dx, Dy, Dz)
        // based on the spherical coordinates (phi, theta).
        // Dx: Component along the X-axis
        // Dy: Component along the Y-axis
        // Dz: Component along the Z-axis
        double Dx = Math.sin(phi) * Math.cos(theta);
        double Dy = Math.sin(phi) * Math.sin(theta);
        double Dz = Math.cos(phi);

        // Iterate through each point (z_idx, x_idx, y_idx) in the 3D array.
        for (int z_idx = 0; z_idx < depth; z_idx++) {
            for (int x_idx = 0; x_idx < width; x_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    // Translate the current point's coordinates so that the center of the volume
                    // becomes the origin (0,0,0).
                    double x_coord = x_idx - centX;
                    double y_coord = y_idx - centY;
                    double z_coord = z_idx - centZ;

                    // Calculate the cross product of two vectors:
                    // 1. Vector P: From the origin (center of volume) to the current point (x_coord, y_coord, z_coord).
                    // 2. Vector D: The cylinder's axis direction vector (Dx, Dy, Dz).
                    // The magnitude of this cross product gives the perpendicular distance from the point P to the line defined by D (passing through the origin).
                    // Cross product C = P x D = (Py*Dz - Pz*Dy, Pz*Dx - Px*Dz, Px*Dy - Py*Dx)
                    double Cx = y_coord * Dz - z_coord * Dy;
                    double Cy = z_coord * Dx - x_coord * Dz;
                    double Cz = x_coord * Dy - y_coord * Dx;

                    // Calculate the squared perpendicular distance from the current point to the cylinder's axis.
                    // Since D is a unit vector, the squared magnitude of the cross product directly gives the squared distance.
                    double distanceSq = Cx * Cx + Cy * Cy + Cz * Cz;

                    // If the squared distance is less than or equal to the squared radius,
                    // the point (z_idx, x_idx, y_idx) is considered to be inside the cylinder.
                    if (distanceSq <= r * r) {
                        env[z_idx][x_idx][y_idx] = 255; // Mark the point as part of the cylinder.
                    }
                }
            }
        }
        return env; // Return the populated 3D array.
    }

    /**
     * Generates a 3D array representing a solid torus. Voxels inside or on the
     * torus surface are valued 255, outside are 0. The torus is centered within
     * the given dimensions.
     *
     * @param depth The depth (z-dimension) of the 3D array.
     * @param width The width (x-dimension) of the 3D array.
     * @param height The height (y-dimension) of the 3D array.
     * @param majorRadius The major radius (R1) of the torus.
     * @param minorRadius The minor radius (r2) of the torus.
     * @return A 3D array of integers [depth][width][height] representing the
     * torus.
     */
    public static int[][][] generateTorus(int depth, int width, int height, double majorRadius, double minorRadius) {
        int[][][] torusVoxels = new int[depth][width][height];

        double centX = width / 2.0;
        double centY = height / 2.0;
        double centZ = depth / 2.0;

        for (int z = 0; z < depth; z++) {
            for (int x = 0; x < width; x++) {
                for (int y = 0; y < height; y++) {
                    double dx = x - centX;
                    double dy = y - centY;
                    double dz = z - centZ;

                    double xy_dist_sq = dx * dx + dy * dy;

                    double left_side_inner_term = xy_dist_sq + dz * dz + majorRadius * majorRadius - minorRadius * minorRadius;

                    double left_side = left_side_inner_term * left_side_inner_term;

                    double right_side = 4.0 * majorRadius * majorRadius * xy_dist_sq;

                    torusVoxels[z][x][y] = left_side <= right_side ? 255 : 0;
                }
            }
        }

        return torusVoxels;
    }
    
    /**
     * Creates a 3D integer array representing a solid torus tilted by specified
     * zenith and azimuth angles. The torus is centered within the 3D volume.
     * Voxels inside or on the torus surface are valued 255, outside are 0.
     *
     * The tilting is achieved by rotating the coordinate system for each voxel
     * before applying the standard torus equation. The 'phi' and 'theta' angles
     * define the direction of the torus's central axis (the axis passing
     * through the hole of the torus).
     *
     * @param depth The depth (size along the Z-axis, first dimension of the array).
     * @param width The width (size along the X-axis, second dimension of the array).
     * @param height The height (size along the Y-axis, third dimension of the array).
     * @param majorRadius The major radius (R) of the torus, defining the distance
     * from the center of the hole to the center of the tube.
     * @param minorRadius The minor radius (r) of the torus, defining the radius
     * of the tube itself.
     * @param phi The zenith angle (polar angle) in radians, measured from the
     * positive Z-axis to the torus's central axis. Ranges from 0 to Math.PI.
     * @param theta The azimuth angle in radians, measured from the positive
     * X-axis in the XY-plane to the projection of the torus's
     * central axis onto the XY-plane. Ranges from 0 to 2 * Math.PI.
     * @return A 3D integer array where 255 indicates a point inside the torus,
     * and 0 otherwise. The array is indexed as env[z_index][x_index][y_index].
     */
     public static int[][][] tiltedTorus(int depth, int width, int height,
                                        double majorRadius, double minorRadius,
                                        double phi, double theta) {

        int[][][] env = new int[depth][width][height];

        double centX = width / 2.0;
        double centY = height / 2.0;
        double centZ = depth / 2.0;

        // Direction cosines of the tilted torus's major axis (new Z-axis in local frame)
        // This vector (Dx, Dy, Dz) represents the direction the torus is "pointing"
        double Dx = Math.sin(phi) * Math.cos(theta);
        double Dy = Math.sin(phi) * Math.sin(theta);
        double Dz = Math.cos(phi);

        for (int z_idx = 0; z_idx < depth; z_idx++) {
            for (int x_idx = 0; x_idx < width; x_idx++) {
                for (int y_idx = 0; y_idx < height; y_idx++) {
                    // Coordinates relative to the center of the volume
                    double x_coord = x_idx - centX;
                    double y_coord = y_idx - centY;
                    double z_coord = z_idx - centZ;

                    // 1. Calculate the coordinate along the tilted axis (z_prime)
                    // This is the projection of the point (x_coord, y_coord, z_coord) onto the (Dx, Dy, Dz) vector.
                    double z_prime = x_coord * Dx + y_coord * Dy + z_coord * Dz;

                    // 2. Calculate the squared distance from the tilted axis (D_sq = x_prime^2 + y_prime^2)
                    // This is derived from the Pythagorean theorem: total_dist_sq = dist_from_axis_sq + z_prime_sq
                    // So, dist_from_axis_sq = total_dist_sq - z_prime_sq
                    double total_dist_sq = x_coord * x_coord + y_coord * y_coord + z_coord * z_coord;
                    double D_sq = total_dist_sq - (z_prime * z_prime);

                    // Apply the implicit equation for a torus in the rotated coordinate system.
                    // The equation is (D_sq + z_prime^2 + R^2 - r^2)^2 = 4 * R^2 * D_sq
                    // where D_sq is (x_prime^2 + y_prime^2) and z_prime is the coordinate along the torus axis.
                    double left_side_inner_term = D_sq + z_prime * z_prime + majorRadius * majorRadius - minorRadius * minorRadius;
                    double left_side = left_side_inner_term * left_side_inner_term;
                    double right_side = 4.0 * majorRadius * majorRadius * D_sq;

                    // If the point satisfies the inequality, it's inside or on the torus
                    if (left_side <= right_side) {
                        env[z_idx][x_idx][y_idx] = 255;
                    } else {
                        env[z_idx][x_idx][y_idx] = 0;
                    }
                }
            }
        }
        return env;
    }

    public static void main(String[] args) {
        // Define pixel values for each point in a grid

        int depth = 50;

        int width = 50;
        int height = 50;

        int[][][] pixelValues = tiltedTorus(depth, width, height, 10, 3, Math.PI/4, Math.PI/4);

        ByteProcessor[] processor = new ByteProcessor[depth];
        ImagePlus[] image = new ImagePlus[depth];

        for (int z = 0; z < depth; z++) {
            processor[z] = new ByteProcessor(width, height);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++) {
                    int val = pixelValues[z][x][y];
                    processor[z].putPixel(x, y, val);
                }

            try {
                String saveTo = "images/input/torus/" + String.format("a%03dtorus", z) + ".png";

                new FileSaver(new ImagePlus("img " + z, processor[z])).saveAsPng(saveTo);
                System.out.println("Image saved as: " + saveTo + ": " + height + "x" + width);

            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    /**
     *
     * TODO: delete this method. It is only for debugging.
     *
     * @brief Reads an image and prints the pixel intensities to the standard
     * output.
     *
     * @param imagePath The path to the image file.
     * @throws IllegalArgumentException If the image cannot be opened or the
     * path is invalid.
     */
    public static void printPixels(File file) {
        // Use the Opener class to open the image
        Opener opener = new Opener();
        ImagePlus image = opener.openImage(file.getPath());

        // Get the ImageProcessor to access pixel data
        ImageProcessor processor = image.getProcessor();

        int width = processor.getWidth();
        int height = processor.getHeight();

        System.out.println("Pixel Intensities:");
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int intensity = processor.getPixel(x, y);
                System.out.print(intensity + " ");
            }
            System.out.println(); // Newline after each row
        }
    }
}
