package fijiPlugin;

import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.IArray1d;
import JCudaWrapper.array.IStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImageJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.HyperStackConverter;
import ij.process.ColorProcessor;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.util.Arrays;
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
public class ImageCreator extends Dimensions {

    /**
     * Array storing color data for each tensor element.
     */
    private final int[] cpuColors;
    private final String[] sliceNames;
    private final int colDist;

    /**
     * Constructs an ImageCreator with the given orientations and coherence
     * tensors.
     *
     * @param handle The GPU computation context.
     * @param sliceNames The names of the slices.
     * @param orientation The tensor representing orientations.
     * @param coherence The tensor representing coherence values. pass null if
     * coherence should not be used
     */
    public ImageCreator(Handle handle, String[] sliceNames, DStrideArray3d orientation, DStrideArray3d coherence) {
        super(handle, orientation);
        colDist = orientation.ld();
        this.sliceNames = sliceNames;
        orientation.setProduct(handle, 2, orientation);

        try (IArray gpuColors = new IStrideArray3d(height, width, depth, batchSize)) {

            int heightCoherence, ldCoherence;
            if (coherence == null) {
                heightCoherence = 0;
                ldCoherence = -1;
            } else {
                heightCoherence = coherence.entriesPerLine();
                ldCoherence = coherence.ld();
            }

            System.out.println("fijiPlugin.ImageCreator.<init>()\n" + orientation.toString());
            
            Kernel.run("colors", handle,
                    orientation.size(),
                    
                    orientation,
                    P.to(orientation.ld()),
                    P.to(orientation.entriesPerLine()),
                    
                    P.to(gpuColors),
                    P.to(gpuColors.ld()),
                    P.to(gpuColors.entriesPerLine()),
                    
                    P.to(coherence),
                    P.to(heightCoherence),
                    P.to(ldCoherence)
            );

            cpuColors = gpuColors.get(handle);           
            
        }

        orientation.setProduct(handle, 0.5, orientation); // Restore original scale.
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
        return cpuColors[frame * width * height * depth + layer * width * height + x * height + y];
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
     * Displays the tensor data as a heat map in Fiji, supporting multiple
     * frames and depths.
     *
     * @param initImageJ Does imageJ need to be initiated.
     */
    public final void printToFiji(boolean initImageJ) {

        if (initImageJ) {
            System.out.println("fijiPlugin.ImageCreator.printToFiji() initiating ImageJ");
            new ImageJ();
        }

        ImageStack stack = new ImageStack(width, height);

        for (int frameIndex = 0; frameIndex < batchSize; frameIndex++) {
            for (int layerIndex = 0; layerIndex < depth; layerIndex++) {
                ColorProcessor cp = new ColorProcessor(width, height);

                for (int x = 0; x < width; x++)
                    for (int y = 0; y < height; y++)
                        cp.set(x, y, getPixelInt(frameIndex, layerIndex, x, y));

                stack.addSlice(sliceNames[frameIndex * depth + layerIndex], cp);
            }
        }

        ImagePlus imp = new ImagePlus("Orientation Heatmap", stack);

        if (depth > 1)
            imp = HyperStackConverter.toHyperStack(
                    imp,
                    1,
                    depth,
                    batchSize
            );
        imp.show();
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

        for (int frame = 0; frame < batchSize; frame++)
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

        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++) {
                getPixelArray(frame, layer, col, row, pixelRGB);
                raster.setPixel(col, row, pixelRGB);
            }

        return image;
    }

    @Override
    public void close() throws Exception {

    }

}
