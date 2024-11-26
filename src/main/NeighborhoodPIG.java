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
import jcuda.Pointer;
import jcuda.runtime.JCuda;

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
     * Writes a heat map orientation picture to the given file.
     *
     * @param writeTo The new orientation image.
     */
    public void orientationColored(String writeTo) {

        try (IArray rgb = stm.getRGB()) {

            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            WritableRaster raster = image.getRaster();

            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++)
                    raster.setPixel(col, row, rgb.get(handle, (col * height + row) * 3, 3));

            try {
                ImageIO.write(image, "png", new File(writeTo));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

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

//        TODO: delete ////////////////////////////////////////
        int d = 400;
        image = image.getSubimage(
                image.getWidth() / 2 - d / 2,
                image.getHeight() / 2 - d / 2 - 1,
                d,
                d + 2
        );
        ////////////////////////////////////////////////

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
//        NeighborhoodPIG np = new NeighborhoodPIG("images/input/debug.jpeg", 1);

        NeighborhoodPIG np = new NeighborhoodPIG("images/input/test.jpeg", 1);

        np.orientationColored("images/output/test.png");

//        System.out.println(np.stm.setOrientations());
    }

}
