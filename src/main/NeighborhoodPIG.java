package main;



import algebra.Matrix;
import algebra.Vector;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import resourceManagement.Handle;
import array.DArray;
import array.DArray2d;
import java.awt.Color;
import java.awt.image.WritableRaster;

/**
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG {

    private StructureTensorMatrix stm;
    private int height, width;
    
    /**
     *
     * @param imagePath The location of the image.
     * @param neighborhoodSize The size of the edges of each neighborhood square.
     * @throws java.io.IOException If there's trouble loading the image.
     */
    public NeighborhoodPIG(String imagePath, int neighborhoodSize) throws IOException {

        Matrix imageMat = processImage(imagePath);

        height = imageMat.getHeight(); 
        width = imageMat.getWidth();
        
        try (Handle handle = new Handle()) {
            
            Gradient grad = new Gradient(imageMat, handle);
            
            stm = new StructureTensorMatrix(grad.getdX(), grad.getdY(), neighborhoodSize);
            
        }
    }
    
    /**
     * Extracts the color at a specific pixel from matrices of colors.
     * @param row
     * @param col
     * @param rgb
     * @return 
     */
    private double [] getColor(int row, int col, double[][][] rgb){
        return new double[]{rgb[0][row][col], rgb[1][row][col], rgb[2][row][col]};
    }
    
    
    public void orientationColored(String writeTo){
        
        double[][][] rgb = stm.getRGB();
                        
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        WritableRaster raster = image.getRaster();
        
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
                raster.setPixel(row, col, getColor(row, col, rgb));
        
        
        // Save the image to a file
        try {
            ImageIO.write(image, "png", new File(writeTo));
        } catch (Exception e) {
            e.printStackTrace();
        }
        
    }

    public static void main(String[] args) throws IOException {
        NeighborhoodPIG np = new NeighborhoodPIG("NematicFiles/images/source/orientation.jpeg", 5);
        
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       /**
     * Interpolates between two colors based on a normalized value in [0, 1].
     * @param start The starting color (corresponding to value 0.0).
     * @param end The ending color (corresponding to value 1.0).
     * @param ratio The normalized value in [0, 1] used to interpolate.
     * @return The interpolated color.
     */
    private static Color interpolateColor(Color start, Color end, double ratio) {
        int red = (int) (start.getRed() + ratio * (end.getRed() - start.getRed()));
        int green = (int) (start.getGreen() + ratio * (end.getGreen() - start.getGreen()));
        int blue = (int) (start.getBlue() + ratio * (end.getBlue() - start.getBlue()));
        
        return new Color(red, green, blue);
    }
    
    /**
     * Method to load a .tif image and convert it into a single-dimensional
     * array in column-major format.
     *
     * @param imagePath The path to the .tif image file
     * @return A matrix of the image data.
     * @throws IOException if the image cannot be loaded
     */
    public final Matrix processImage(String imagePath) throws IOException {

        BufferedImage image = ImageIO.read(new File(imagePath));

        if (image.getType() != BufferedImage.TYPE_BYTE_GRAY)
            image = convertToGrayscale(image);

        Raster raster = image.getRaster();
        int width = image.getWidth(), height = image.getHeight();

        double[] imageData = new double[width * height];

        Arrays.setAll(imageData, i -> raster.getSample(i / height, i % height, 0) / 255.0);

        
        
        try (Handle handle = new Handle()) {
            
            DArray data = new DArray(handle, imageData);
            
            return new Matrix(
                    handle, 
                    data,
                    height,
                    width);
        }
    }

    /**
     * Converts a given BufferedImage to grayscale.
     *
     * @param image The original BufferedImage
     * @return A grayscale BufferedImage
     */
    private BufferedImage convertToGrayscale(BufferedImage image) {

        BufferedImage grayImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);

        Graphics2D g2d = grayImage.createGraphics();
        g2d.drawImage(image, 0, 0, null);
        g2d.dispose();

        return grayImage; // Return the grayscale image
    }

}
