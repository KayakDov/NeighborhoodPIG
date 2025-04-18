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

    public static void main(String[] args) {
        // Define pixel values for each point in a grid

        int depth = 1;
        int[][] pixelValues = blackWithWhiteBorderY(19, 6, 2);

        int width = pixelValues[0].length;
        int height = pixelValues.length;

        // Create an ImageProcessor for an 8-bit grayscale image
        ImageProcessor processor = new ByteProcessor(width, height);

        // Set pixel values in the ImageProcessor
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int grayValue = pixelValues[y][x];
                processor.putPixel(x, y, grayValue);
            }
        }

        // Create an ImagePlus object
        ImagePlus image = new ImagePlus("Debug Image", processor);

        // Save the image as a JPEG file
        try {
            String saveTo = "images/input/debug/" + depth + "debugWhite.png";

            FileSaver saver = new FileSaver(image);
            saver.saveAsPng(saveTo);
            System.out.println("Image saved as: " + saveTo + ": " + height + "x" + width);

//            printPixels(new File(saveTo));
        } catch (Exception e) {
            e.printStackTrace();
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
