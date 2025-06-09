package imageWork;

import fijiPlugin.Dimensions;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.FileSaver;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.LUT;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 *
 * @author E. Dov Neimand
 */
public class MyImagePlus extends ImagePlus {

    private Dimensions dim;

    /**
     * Sets the image.
     *
     * @param title The title.
     * @param imp The stack.
     */
    public MyImagePlus(String title, ImageStack imp, int depth) {
        super(title, imp);
        dim = new Dimensions(imp, depth);
    }

    /**
     * Converts a float32 grayscale ImagePlus to a normalized 8-bit grayscale
     * ImagePlus. This is useful for saving or viewing scientific float images
     * in standard image viewers. This method correctly handles
     * multi-slice/multi-frame ImagePlus objects by processing each slice's
     * processor individually.
     *
     * @return A new ImagePlus with pixel values scaled to the 8-bit range [0,
     * 255].
     */
    public ImagePlus normalizeFloatImageTo8Bit() {

        ImageStack byteStack = new ImageStack(getWidth(), getHeight());
        int totalSlices = getStackSize();

        for (int i = 1; i <= totalSlices; i++) {
            setSlice(i);
            ImageProcessor currentFloatProcessor = getProcessor(); // Get its processor

            if (!(currentFloatProcessor instanceof FloatProcessor))
                throw new IllegalArgumentException("Slice " + i + " is not a FloatProcessor.");

            FloatProcessor fp = (FloatProcessor) currentFloatProcessor;

            double min = getDisplayRangeMin();
            double max = getDisplayRangeMax();

            fp.setMinAndMax(min, max);
            ByteProcessor bp = fp.convertToByteProcessor();

            byteStack.addSlice(getStack().getSliceLabel(i), bp);
        }

        ImagePlus normalizedImp = new ImagePlus(getTitle() + "_8bit", byteStack);

        if (isHyperStack()) {
            normalizedImp.setDimensions(
                    getNChannels(),
                    dim.depth,
                    dim.batchSize
            );
        }

        return normalizedImp;
    }

    /**
     * Prepares the image for saving. If the image is a 32-bit float, it is
     * normalized to 8-bit. Otherwise, a copy of the original image is created.
     * This makes the saveSlices method more robust for different image types.
     *
     * @return An ImagePlus ready for saving.
     */
    private ImagePlus prepareImageForSaving() {
        if (getType() == ImagePlus.GRAY32) {
            return normalizeFloatImageTo8Bit();
        } else {
            // For non-float images, we can work with a copy directly.
            return duplicate();
        }
    }

    /**
     * Saves each slice (channel, Z-slice, and T-frame) of an ImagePlus to a
     * separate PNG file. This method handles file paths that contain
     * subdirectories within the image title by creating them as needed.
     *
     * @param saveTo The base folder where the files should be saved.
     */
    public void saveSlices(String saveTo) {
        // **FIX:** Get the original title, strip the path to get just the filename,
        // and remove any leading underscore.
        String fileNameOnly = new File(getTitle()).getName();
        String baseFileName = fileNameOnly.startsWith("_") ? fileNameOnly.substring(1) : fileNameOnly;

        ImagePlus impToSave = prepareImageForSaving();

        int numSlicesZ = impToSave.getNSlices();
        int numFrames = impToSave.getNFrames();
        int numChannels = impToSave.getNChannels();

        for (int frame = 1; frame <= numFrames; frame++) {
            for (int sliceZ = 1; sliceZ <= numSlicesZ; sliceZ++) {
                for (int channel = 1; channel <= numChannels; channel++) {
                    int linearStackIndex = impToSave.getStackIndex(channel, sliceZ, frame);
                    impToSave.setSlice(linearStackIndex);

                    ImageProcessor currentProcessor = impToSave.getProcessor();
                    ImagePlus sliceToSave = new ImagePlus(
                            baseFileName + "_C" + channel + "_F" + frame + "_Z" + sliceZ,
                            currentProcessor
                    );

                    FileSaver fs = new FileSaver(sliceToSave);
                    String specificFileName = sliceToSave.getTitle() + ".png";
                    File outputFile = new File(saveTo, specificFileName);

                    // Ensure the full directory path exists before saving.
                    File parentDir = outputFile.getParentFile();
                    if (parentDir != null && !parentDir.exists()) {
                        if (!parentDir.mkdirs()) {
                            System.err.println("Failed to create directory: " + parentDir.getAbsolutePath());
                            continue; // Skip to the next slice if directory creation fails
                        }
                    }

                    String fullPath = outputFile.getAbsolutePath();
                    try {
                        if (!fs.saveAsPng(fullPath)) {
                            System.err.println("Failed to save png image: " + fullPath);
                        } else {
                            System.out.println("imageWork.ImgPlsToFiles.saveSlices - Saved: " + fullPath);
                        }
                    } catch (Exception e) {
                        System.err.println("Exception while saving " + fullPath + ": " + e.getMessage());
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    /**
     * Prints the pixel values of a given ImagePlus object to standard output.
     * The values are printed as floats. For multi-slice images, it iterates
     * through each slice (z-dimension). For multi-channel images, it iterates
     * through each channel. For multi-frame images, it iterates through each
     * frame (t-dimension).
     *
     */
    public void stdOutputImageValues() {

        new RuntimeException().printStackTrace(System.out);

        System.out.println("--- Image Details ---");
        System.out.println("Title: " + getTitle() + "\n" + dim.toString());
        System.out.println("---------------------");

        for (int t = 1; t <= dim.batchSize; t++) {
            for (int z = 1; z <= dim.depth; z++) {

                setPosition(1, z, t);
                ImageProcessor ip = getProcessor();

                System.out.println("\n--- Values for Slice " + z + ", Frame " + t + " ---");
                for (int y = 0; y < dim.height; y++) {
                    for (int x = 0; x < dim.width; x++) {
                        float value = ip.getf(x, y);
                        System.out.printf("%8.1f ", value);
                    }
                    System.out.println();
                }
            }

        }
        System.out.println("\n--- End of Image Values ---");
    }

    /**
     * Superimposes a binary stack onto the current grayscale image using the
     * specified color.
     *
     * @param overlayStack The binary stack (0=transparent, 255=opaque).
     * @param color The color to render the opaque (255) regions in.
     * @return this
     */
    public MyImagePlus overlayBinaryMask(ImageStack overlayStack, Color color) {
        Overlay overlay = new Overlay();

        for (int z = 1; z <= overlayStack.getSize(); z++) {
            ByteProcessor binaryProcessor = (ByteProcessor) overlayStack.getProcessor(z);

            // Create a color image from binary mask
            ColorProcessor colorProcessor = new ColorProcessor(width, height);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    if (binaryProcessor.get(x, y) == 255) colorProcessor.set(x, y, color.getRGB());
                    else colorProcessor.set(x, y, 0); // transparent black

            ImageRoi roi = new ImageRoi(0, 0, colorProcessor);
            roi.setZeroTransparent(true); // make zero pixels transparent
            roi.setPosition(z); // position in the stack

            overlay.add(roi);
        }
        setOverlay(overlay);
        return this;
    }
}
