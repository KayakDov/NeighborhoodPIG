package main;

import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.Arrays;
import java.util.Random;
import javax.imageio.ImageIO;

/**
 * This file generates a jpeg to use for debugging purposes.
 *
 * @author dov
 */
public class GenDebugFile {

    
    public static double[][] xGrad1 = new double[][]{
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
            {0.1, 0.2, 0.3, 0.4, 0.5, 0.6},
        };
    
    public static double[][] white = new double[][]{
        {1,1,1,1,1,1},
        {1,1,1,1,1,1},
        {1,1,1,1,1,1},
        {1,1,1,1,1,1},
        {1,1,1,1,1,1},
        {1,1,1,1,1,1},
        {1,1,1,1,1,1}
    };
    
    
    public static double[][] random(int height, int width){
        Random rand = new Random();
        double[][] random = new double[height][width];
        for(double[] row: random)
            Arrays.setAll(row, i -> rand.nextDouble());
        return random;
    }
    public static void main(String[] args) {
        
        // Define pixel values for each point in a 7x7 grid
        // We use values from 0.1, 0.2, ... to 0.7
        double[][] pixelValues = xGrad1;

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
            String saveTo = "images/input/debug.jpeg";
            File outputFile = new File(saveTo);
            ImageIO.write(image, "jpeg", outputFile);
            System.out.println("Image saved as:" + saveTo + ": "+ height + "x" + width);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
