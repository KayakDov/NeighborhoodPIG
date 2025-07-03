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

    public static void main(String[] args) {
        // Define pixel values for each point in a grid

        int depth = 15;

        int width = 60;
        int height = 60;

        int[][][] pixelValues = generateTorus(depth, width, height, 20, 3);

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
                String saveTo = "images/input/torus/" + String.format("%03d", z) + ".png";

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
