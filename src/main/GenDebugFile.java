package main;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import javax.imageio.ImageIO;

/**
 * This file generates a jpeg to use for debugging purposes.
 *
 * @author dov
 */
public class GenDebugFile {

    public static void main(String[] args) {
        
        // Define pixel values for each point in a 7x7 grid
        // We use values from 0.1, 0.2, ... to 0.7
        double[][] pixelValues = {
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
        };

        int width = pixelValues[0].length;
        int height = pixelValues.length;
        
        // Create a grayscale BufferedImage
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster raster = image.getRaster();

        // Set each pixel to a grayscale value based on pixelValues
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Scale the pixel value to the range 0-255
                int grayValue = (int) (pixelValues[y][x] * 255);
                raster.setSample(x, y, 0, grayValue);
            }
        }

        // Save the image as a JPEG file
        try {
            File outputFile = new File("images/input/debug.jpeg");
            ImageIO.write(image, "jpeg", outputFile);
            System.out.println("Image saved as images/input/debug.jpeg, a " + height + "x" + width + " image.");
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
