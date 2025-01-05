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

    
    /**
     * A heat map of the orientation in the xy plane.
     * @return A heat map of the orientation in the xy plane.
     */
    public ImageCreator getImageOrientationXY(){
        return new ImageCreator(handle, stm.getOrientationXY(), stm.getCoherence());
    }
    
    
    /**
     * A heat map of the orientation in the yz plane.
     * @return A heat map of the orientation in the yz plane.
     */
    public ImageCreator getImageOrientationYZ(){
        return new ImageCreator(handle, stm.getOrientationXY(), stm.getCoherence());
    }
    
    @Override
    public void close() {
        stm.close();
        handle.close();
    }

}
