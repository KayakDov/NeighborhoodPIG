package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.frame.ColorPicker;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.image.BufferedImage;
import java.io.File;
import javax.imageio.ImageIO;
import java.awt.image.WritableRaster;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG implements AutoCloseable {

    private StructureTensorMatrix stm;
    private int height, width, depth, duration;
    private Handle handle;

    public static boolean D3 = true, D2 = false;

    /**
     *
     * @param imp The image from fiji.
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @param tolerance How close must a number be to 0 to be considered 0.
     */
    public NeighborhoodPIG(ImagePlus imp, int neighborhoodSize, double tolerance) {

        handle = new Handle();

        width = imp.getWidth();
        height = imp.getHeight();
        depth = imp.getNSlices() / imp.getNFrames();
        duration = imp.getNFrames();

        TensorOrd3Stride image = new TensorOrd3Stride(handle, height, width, depth, duration, processImage(imp));

        Gradient grad = new Gradient(image, handle);

        image.close();
        stm = new StructureTensorMatrix(grad, neighborhoodSize, tolerance);
        grad.close();
    }

    /**
     * Converts a grayscale ImagePlus object into a single column-major gpu
     * array of pixel values.
     *
     * @param imp The input grayscale ImagePlus object.
     * @return A single column-major gpu array containing pixel values of all
     * slices.
     * @throws IllegalArgumentException If the input image is not grayscale.
     */
    /**
     * Converts a grayscale ImagePlus object into a single column-major GPU
     * array of pixel values.
     *
     * @param imp The input grayscale ImagePlus object.
     * @return A single column-major GPU array containing pixel values of all
     * slices.
     * @throws IllegalArgumentException If the input image is not grayscale.
     */
    public final DArray processImage(ImagePlus imp) {

        // Ensure the input image stack is converted to 32-bit float
        ImageStack stack = imp.getStack();
        double[] columnMajorArray = new double[width * height * depth * duration];
        int index = 0;

        for (int slice = 1; slice <= stack.getSize(); slice++) {
            // Convert to FloatProcessor if not already one
            ImageProcessor ip = stack.getProcessor(slice);
            if (!(ip instanceof FloatProcessor)) {
                ip = ip.convertToFloat();
            }
            float[] pixels = (float[]) ip.getPixels();

            for (int col = 0; col < width; col++) {
                for (int row = 0; row < height; row++) {
                    columnMajorArray[index++] = pixels[row * width + col];
                }
            }
        }

        return new DArray(handle, columnMajorArray);
    }

//    /**
//     * Writes a heat map orientation picture to the given file.
//     *
//     * @param writeTo The new orientation image.
//     */
//    public void orientationColored(String writeTo) {
//
//        try (IArray rgb = stm.getRGBs(false)) {
//
//            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
//            WritableRaster raster = image.getRaster();
//
//            int[] cpuRGB = rgb.get(handle);
//            int[] pixelRGB = new int[3];
//            for (int row = 0; row < height; row++)
//                for (int col = 0; col < width; col++) {
//                    System.arraycopy(cpuRGB, (col * height + row) * 3, pixelRGB, 0, 3);
//                    raster.setPixel(col, row, pixelRGB);
//                }
//
//            try {
//                ImageIO.write(image, "png", new File(writeTo));
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
//        }
//
//    }
    /**
     * Processes the given ImagePlus object and converts it into a 3D tensor
     * with GPU strides.
     */
    public final void fijiDisplayOrientationHeatmap() {

        if ((long) width * height * depth * duration > Integer.MAX_VALUE)
            throw new IllegalArgumentException("Image size exceeds array limit.");

        ImageStack stack = new ImageStack(width, height);

        int layerSize = height * width;

        StructureTensorMatrix.RGB rgb = stm.getRGB();

        for (int frameIndex = 0; frameIndex < duration; frameIndex++)
            for (int layerIndex = 0; layerIndex < depth; layerIndex++) {

                ColorProcessor cp = new ColorProcessor(width, height);
                for (int x = 0; x < width; x++) for (int y = 0; y < height; y++)
                        cp.set(x, y, rgb.getPixel(frameIndex, layerIndex, x, y));
                stack.addSlice("Frame " + frameIndex + " depth " + layerIndex, cp);
            }

        ImagePlus imagePlus = new ImagePlus("Orientation Colored Images", stack);

        imagePlus.show();
    }

    @Override
    public void close() {
        stm.close();
        handle.close();
    }

}
