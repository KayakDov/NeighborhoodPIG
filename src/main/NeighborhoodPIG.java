package main;

import JCudaWrapper.algebra.Matrix;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.array.DArray;
import java.awt.image.WritableRaster;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG implements AutoCloseable {

    private StructureTensorMatrix stm;
    private int height, width;
    private Handle handle;

    /**
     *
     * @param imagePath The location of the image.
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @throws java.io.IOException If there's trouble loading the image.
     */
    public NeighborhoodPIG(String imagePath, int neighborhoodSize) throws IOException {
        handle = new Handle();

        Matrix imageMat = processImage(imagePath, handle);
        Gradient grad = new Gradient(imageMat, handle);
        imageMat.close();
        stm = new StructureTensorMatrix(grad.x(), grad.y(), neighborhoodSize);
        grad.close();

    }

    /**
     * Extracts the color at a specific pixel from matrices of colors.
     *
     * @param row
     * @param col
     * @param rgb
     * @return
     */
    private double[] getColor(int row, int col, double[][][] rgb) {
        return new double[]{rgb[0][row][col], rgb[1][row][col], rgb[2][row][col]};
    }

    /**
     * Writes a heat map orientation picture to the given file.
     *
     * @param writeTo The new orientation image.
     */
    public void orientationColored(String writeTo) {

        double[][][] rgb = stm.getRGB();

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        WritableRaster raster = image.getRaster();

        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++)
                raster.setPixel(col, row, getColor(row, col, rgb));

        // Save the image to a file
        try {
            ImageIO.write(image, "png", new File(writeTo));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void main(String[] args) throws IOException {
//        NeighborhoodPIG np = new NeighborhoodPIG("images/input/debug.jpeg", 1);
        NeighborhoodPIG np = new NeighborhoodPIG("images/input/test.jpeg", 1);
        np.orientationColored("images/output/test.png");

//        System.out.println(np.stm.setOrientations());
    }

    /**
     * Method to load a .tif image and convert it into a single-dimensional
     * array in column-major format.
     *
     * @param imagePath The path to the .tif image file
     * @param handle
     * @return A matrix of the image data.
     * @throws IOException if the image cannot be loaded
     */
    public final Matrix processImage(String imagePath, Handle handle) throws IOException {

        BufferedImage image = ImageIO.read(new File(imagePath));

        if (image.getType() != BufferedImage.TYPE_BYTE_GRAY)
            image = convertToGrayscale(image);

        Raster raster = image.getRaster();
        int size = (int) 1e5;
        width = Math.min(size, image.getWidth());
        height = Math.min(size, image.getHeight());//TODO: change to full size by removing the size variabl and the min.

        double[] imageData = new double[width * height];

        Arrays.setAll(imageData, i -> raster.getSample(i / height, i % height, 0) / 255.0);

        Matrix mat = new Matrix(
                handle,
                new DArray(handle, imageData),
                height,
                width);

//        System.out.println(mat);
        return mat;
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

    @Override
    public void close() {
        stm.close();
        handle.close();
    }

}
