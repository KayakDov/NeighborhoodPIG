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
import JCudaWrapper.array.IArray;
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

    public static boolean D3 = true, D2 = false;
    
    /**
     *
     * @param imagePath The location of the image.
     * @param neighborhoodRad The radius of the edges of each neighborhood
     * square.
     * @param is3d true for 3d, false for 2d.
     * @param tolerance How close must a number be to 0 to be considered 0.
     * @throws java.io.IOException If there's trouble loading the image.
     */
    public NeighborhoodPIG(String imagePath, int neighborhoodRad, boolean is3d, double tolerance) throws IOException {

        handle = new Handle();

        Matrix imageMat = processImage(imagePath, handle);

        Gradient grad = new Gradient(imageMat, handle);

        imageMat.close();
        stm = new StructureTensorMatrix(grad.x(), grad.y(), neighborhoodRad, tolerance);
        grad.close();

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

        width = image.getWidth();
        height = image.getHeight();

        double[] imageDataCPU = new double[width * height];

        Arrays.setAll(imageDataCPU, i -> raster.getSample(i / height, i % height, 0) / 255.0);

        DArray imageDataGPU = new DArray(handle, imageDataCPU);

        Matrix mat = new Matrix(
                handle,
                imageDataGPU,
                height,
                width);

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

    public static void main(String[] args) throws IOException {

//        try (NeighborhoodPIG np = new NeighborhoodPIG("images/input/debug.jpeg", 10, D2, 1e-11)) {
//            np.orientationColored("images/output/test.png");
//        }
        
        try (NeighborhoodPIG np = new NeighborhoodPIG("images/input/test.jpeg", 10, D2, 1e-11)) {
//            np.orientationColored("images/output/test.png");
        }
    }

}
