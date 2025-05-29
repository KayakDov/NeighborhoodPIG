package imageWork;

import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToI2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ColorProcessor;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import javax.imageio.ImageIO;

/**
 * A class for creating orientation heatmaps and saving them as images.
 *
 * This class processes 3D tensor data to generate color-coded orientation maps.
 * It supports GPU-accelerated processing and provides functionality for
 * displaying and saving images.
 *
 * @author E. Dov Neimand
 */
public class ColorHeatMapCreator extends HeatMapCreator implements AutoCloseable{

    protected final PArray2dToI2d colors;

    /**
     *
     * @param handle
     * @param sliceNames The names of the slices.
     * @param stackName The name of the stack.
     * @param orientation The dimensions.
     * @param coherence The intensity of each color.
     * @param dim The dimensions.
     */
    public ColorHeatMapCreator(Handle handle, String[] sliceNames, String stackName, PArray2dToF2d orientation, PArray2dToF2d coherence, Dimensions dim) {
        super(sliceNames, stackName, dim, handle);
        orientation.scale(handle, 2);

        colors = new PArray2dToI2d(dim.depth, dim.batchSize, dim.height, dim.width);
        colors.initTargets(handle);

        Kernel.run("colors", handle,
                dim.size(),
                new PArray2dTo2d[]{orientation, colors, coherence},
                dim
        );
        
        orientation.scale(handle, 0.5);

    }

    /**
     * Retrieves the color for a specific coordinate.
     *
     * @param frame The frame index.
     * @param layer The layer index.
     * @param x The x-coordinate.
     * @param y The y-coordinate.
     * @return The color as an integer in RGB format.
     */
    private int getPixelInt(int[] layer, int x, int y) {
        return layer[x * dim.height + y];
    }

    /**
     * Extracts the RGB components of the color for a specific coordinate.
     *
     * @param frame The frame index.
     * @param layer The layer index.
     * @param x The x-coordinate.
     * @param y The y-coordinate.
     * @param pixelRGBHere An array to store the RGB values.
     * @return The same array with R, G, and B values populated.
     */
    private int[] getPixelArray(int[] layer, int x, int y, int[] pixelRGBHere) {
        int rgbInt = getPixelInt(layer, x, y);
        pixelRGBHere[0] = (rgbInt >> 16) & 0xFF; // Red
        pixelRGBHere[1] = (rgbInt >> 8) & 0xFF;  // Green
        pixelRGBHere[2] = rgbInt & 0xFF;         // Blue
        return pixelRGBHere;
    }

    /**
     * Displays the tensor data as a heat map in Fiji, supporting multiple
     * frames and depths.
     */
    @Override
    public final void printToFiji() {

        int[] layer = new int[dim.layerSize()];
        
        ImageStack stack = dim.getImageStack();

        for (int frameIndex = 0; frameIndex < dim.batchSize; frameIndex++) {
            for (int layerIndex = 0; layerIndex < dim.depth; layerIndex++) {
                
                setLayer(layerIndex, frameIndex, layer);
                
                ColorProcessor cp = new ColorProcessor(dim.width, dim.height);

                for (int x = 0; x < dim.width; x++)
                    for (int y = 0; y < dim.height; y++)
                        cp.set(x, y, getPixelInt(layer, x, y));

                stack.addSlice(sliceNames[frameIndex * dim.depth + layerIndex], cp);
            }
        }

        ImagePlus imp = new ImagePlus(stackName, stack);

        dim.setToHyperStack(imp).show();
    }

    /**
     * Loads the proffered array from the gpu.
     * @param layerInd The index of the desired layer.
     * @param frameInd The index of the desired frame.
     * @param layer Column major array where the values are to be stored.
     */
    private void setLayer(int layerInd, int frameInd, int[] layer){
        colors.get(layerInd, frameInd).getVal(handle).get(handle, layer);
    }
    
    /**
     * Saves orientation heatmaps as images in the specified folder.
     *
     * @param writeToFolder The folder where images will be saved.
     */
    @Override
    public void printToFile(String writeToFolder) {
        // Ensure the folder exists, create it if it doesn't.
        File directory = new File(writeToFolder);
        if (!directory.exists()) directory.mkdirs();

        int[] pixelRGB = new int[3];
        int[] layer = new int[dim.layerSize()];

        for (int frameInd = 0; frameInd < dim.batchSize; frameInd++) {
            for (int layerInd = 0; layerInd < dim.depth; layerInd++) {
                
                setLayer(layerInd, frameInd, layer);
                
                BufferedImage image = createImage(layer, pixelRGB);

                String fileName = sliceNames[frameInd * dim.depth + layerInd],
                        fileType = fileName.substring(fileName.lastIndexOf('.') + 1);

                File outputFile = new File(writeToFolder, fileName);

                try {
                    ImageIO.write(image, fileType, outputFile);
                    System.out.println("Image printed to " + outputFile);
                } catch (Exception e) {
                    System.err.println("Error writing image to file: " + outputFile.getAbsolutePath());
                    e.printStackTrace();
                }
            }
        }
    }

    /**
     * Creates a BufferedImage for a specific frame and layer.
     *
     * @param frame The frame index.
     * @param layer The layer index.
     * @param pixelRGB Array to store RGB values.
     * @return A BufferedImage representing the heatmap.
     */
    private BufferedImage createImage(int[] layer, int[] pixelRGB) {
        BufferedImage image = new BufferedImage(dim.width, dim.height, BufferedImage.TYPE_INT_RGB);
        WritableRaster raster = image.getRaster();

        for (int row = 0; row < dim.height; row++) {
            for (int col = 0; col < dim.width; col++) {
                getPixelArray(layer, col, row, pixelRGB);
                raster.setPixel(col, row, pixelRGB);
            }
        }

        return image;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        colors.close();
    }

}
