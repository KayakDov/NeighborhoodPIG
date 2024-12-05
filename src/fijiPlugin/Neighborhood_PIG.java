package fijiPlugin;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.TensorOrd3;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;
import ij.plugin.PlugIn;
import java.awt.image.WritableRaster;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class Neighborhood_PIG implements AutoCloseable, PlugIn {

    private StructureTensorMatrix stm;
    private int height, width;
    private Handle handle;

    public static boolean D3 = true, D2 = false;
    
    /**
     *
     * @param imageMat
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @param is3d true for 3d, false for 2d.
     * @param tolerance How close must a number be to 0 to be considered 0.
     * @throws java.io.IOException If there's trouble loading the image.
     */
    public Neighborhood_PIG(TensorOrd3 imageMat, int neighborhoodSize, boolean is3d, double tolerance) throws IOException {

//        handle = imageMat.getHandle();        
//
//        Gradient grad = new Gradient(imageMat, handle);
//
//        imageMat.close();
//        stm = new StructureTensorMatrix(grad.x(), grad.y(), neighborhoodSize, tolerance);
//        grad.close();

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

            int[] cpuRGB = rgb.get(handle);
            int[] pixelRGB = new int[3];
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++){
                    System.arraycopy(cpuRGB, (col * height + row) * 3, pixelRGB, 0, 3);
                    raster.setPixel(col, row, pixelRGB);
                }

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

    public static void main(String[] args) {
        
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

//    public static void main(String[] args) throws IOException {
//
////        try (NeighborhoodPIG np = new NeighborhoodPIG("images/input/debug.jpeg", 10, D2, 1e-11)) {
////            np.orientationColored("images/output/test.png");
////        }
//        
//        try (Neighborhood_PIG np = new Neighborhood_PIG("images/input/test.jpeg", 10, D2, 1e-11)) {
//            np.orientationColored("images/output/test.png");
//        }
//    }

    @Override
    public void run(String string) {
        
        IJ.showMessage("Window title", "Hello from Neighborhood_PIG!");
        
//        ImagePlus imp = ij.WindowManager.getCurrentImage();
//        if (imp == null) {
//            ij.IJ.showMessage("No image open.");
//            return;
//        }
//        
//        int width = imp.getWidth();
//        int height = imp.getHeight();
//        int depth = imp.getStackSize();
//        
//        
//        ij.IJ.showMessage("There is an open image.");
//        for (int z = 1; z <= depth; z++) {
//            ImageProcessor ip = imp.getStack().getProcessor(z);
//            
//            // Iterate through each pixel in the slice
//            for (int y = 0; y < height; y++) {
//                for (int x = 0; x < width; x++) {
//                    // Check if the pixel is "every other" pixel based on a pattern (x + y)
//                    if ((x + y) % 2 == 0) {
//                        // Change pixel color to blue (in RGB format)
//                        ip.putPixel(x, y, (255 << 16) | (255 << 8)); // Blue with full intensity, red and green are zero
//                    }
//                }
//            }
//        }
        
        // Update the image to reflect changes
//        imp.updateAndDraw();
    }

}
