package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3StrideDim;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
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
public class ImageCreator extends TensorOrd3StrideDim {

    /**
     * Array storing color data for each tensor element.
     */
    private final int[] cpuColors;
    private final String[] sliceNames;

    /**
     * Constructs an ImageCreator with the given orientations and coherence
     * tensors.
     *
     * @param handle The GPU computation context.
     * @param sliceNames The names of the slices.
     * @param orientation The tensor representing orientations.
     * @param coherence The tensor representing coherence values. pass null if coherence should not be used
     */
    public ImageCreator(Handle handle, String[] sliceNames, TensorOrd3Stride orientation, TensorOrd3Stride coherence) {
        super(orientation);
        this.sliceNames = sliceNames;
        orientation.dArray().multiply(handle, 2, 1); // Scale orientations.

        try (IArray gpuColors = IArray.empty(orientation.dArray().length)) {

            Kernel.run("colors", handle,
                    orientation.size(),
                    orientation.dArray(),
                    P.to(1),
                    P.to(gpuColors),
                    P.to(1),
                    P.to(coherence == null ? new TensorOrd3Stride(handle, 0, 0, 0, 0):coherence),
                    P.to(coherence == null?-1:1)
            );
            cpuColors = gpuColors.get(handle); // Transfer GPU results to CPU.
        }

        orientation.dArray().multiply(handle, 0.5, 1); // Restore original scale.
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
    private int getPixelInt(int frame, int layer, int x, int y) {
        return cpuColors[frame * layerDist * depth + layer * layerDist + x * colDist + y];
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
    private int[] getPixelArray(int frame, int layer, int x, int y, int[] pixelRGBHere) {
        int rgbInt = getPixelInt(frame, layer, x, y);
        pixelRGBHere[0] = (rgbInt >> 16) & 0xFF; // Red
        pixelRGBHere[1] = (rgbInt >> 8) & 0xFF;  // Green
        pixelRGBHere[2] = rgbInt & 0xFF;         // Blue
        return pixelRGBHere;
    }

    /**
     * Displays the tensor data as a heatmap in Fiji.
     */
    public final void printToFiji() {

        ImageStack frames = new ImageStack(width, height);

        for (int frameIndex = 0; frameIndex < batchSize; frameIndex++) {
            ImageStack layers = new ImageStack(width, height);
            for (int layerIndex = 0; layerIndex < depth; layerIndex++) {
                ColorProcessor cp = new ColorProcessor(width, height);
                for (int x = 0; x < width; x++)
                    for (int y = 0; y < height; y++)
                        cp.set(x, y, getPixelInt(frameIndex, layerIndex, x, y));

                layers.addSlice(sliceNames[frameIndex * depth + layerIndex], cp);
            }
            frames.addSlice("frame " + frameIndex, layers.getProcessor(1)); // Add the completed frame
        }

        new ImagePlus("Orientation Heatmap", frames).show();
    }

        /**
         * Saves orientation heatmaps as images in the specified folder.
         *
         * @param writeToFolder The folder where images will be saved.
         */
    public void printToFile(String writeToFolder) {
        // Ensure the folder exists, create it if it doesn't.
        File directory = new File(writeToFolder);
        if (!directory.exists()) directory.mkdirs();

        int[] pixelRGB = new int[3];

        for (int frame = 0; frame < batchSize; frame++) {
            for (int layer = 0; layer < depth; layer++) {
                BufferedImage image = createImage(frame, layer, pixelRGB);

                String fileName = sliceNames[frame * depth + layer],
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
    private BufferedImage createImage(int frame, int layer, int[] pixelRGB) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        WritableRaster raster = image.getRaster();

        for (int row = 0; row < height; row++) {
            for (int col = 0; col < width; col++) {
                getPixelArray(frame, layer, col, row, pixelRGB);
                raster.setPixel(col, row, pixelRGB);
            }
        }

        return image;
    }
}
