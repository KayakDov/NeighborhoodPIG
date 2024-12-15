package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ColorProcessor;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import ij.process.ImageProcessor;
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

        TensorOrd3Stride image = processImage(imp);

        Gradient grad = new Gradient(imageMat, handle);
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

        try (IArray rgb = stm.getRGBArray()) {

            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            WritableRaster raster = image.getRaster();

            int[] cpuRGB = rgb.get(handle);
            int[] pixelRGB = new int[3];
            for (int row = 0; row < height; row++)
                for (int col = 0; col < width; col++) {
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
     * Processes the given ImagePlus object and converts it into a 3D tensor
     * with GPU strides.
     */
    public final void fijiDisplayOrientationHeatmap() {

        ImageStack stack = new ImageStack(width, height);

        try (IArray gpuColors = stm.getRGBs()) {

            if ((long) width * height * depth * duration > Integer.MAX_VALUE)
                throw new IllegalArgumentException("Image size exceeds array limit.");

            int[] colorsCPU = gpuColors.get(handle, 0, gpuColors.length);
            int[] slice = new int[height * width];
            int colorsIndex = 0, layerSize = height*width;

            for (int frameIndex = 0; frameIndex < duration; frameIndex++)
                for (int layerIndex = 0; layerIndex < depth; layerIndex++) {
                    System.arraycopy(colorsCPU, frameIndex*layerIndex*layerSize, slice, 0, layerSize);
                    ColorProcessor sliceProcessor = new ColorProcessor(width, height, slice);
                    stack.addSlice("Frame " + frameIndex + " depth " + layerIndex, sliceProcessor);
                }
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
