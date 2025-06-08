package imageWork;

import fijiPlugin.Dimensions;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 *
 * @author E. Dov Neimand
 */
public class ImagePlusUtil {

    private ImagePlus imp;
    private Dimensions dim;

    /**
     * Sets the image.
     *
     * @param imp
     */
    public ImagePlusUtil(ImagePlus imp) {
        this.imp = imp;
        dim = new Dimensions(imp);
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

        ImageStack byteStack = new ImageStack(imp.getWidth(), imp.getHeight());
        int totalSlices = imp.getStackSize();

        for (int i = 1; i <= totalSlices; i++) {
            imp.setSlice(i);
            ImageProcessor currentFloatProcessor = imp.getProcessor(); // Get its processor

            if (!(currentFloatProcessor instanceof FloatProcessor))
                throw new IllegalArgumentException("Slice " + i + " is not a FloatProcessor.");

            FloatProcessor fp = (FloatProcessor) currentFloatProcessor;


            double min = imp.getDisplayRangeMin();
            double max = imp.getDisplayRangeMax();

            fp.setMinAndMax(min, max);
            ByteProcessor bp = fp.convertToByteProcessor();

            byteStack.addSlice(imp.getStack().getSliceLabel(i), bp);
        }

        ImagePlus normalizedImp = new ImagePlus(imp.getTitle() + "_8bit", byteStack);

        if (imp.isHyperStack()) {
            normalizedImp.setDimensions(
                    imp.getNChannels(),
                    imp.getNSlices(),
                    imp.getNFrames()
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
        if (imp.getType() == ImagePlus.GRAY32) {
            return normalizeFloatImageTo8Bit();
        } else {
            // For non-float images, we can work with a copy directly.
            return imp.duplicate();
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
        String fileNameOnly = new File(this.imp.getTitle()).getName();
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
        System.out.println("Title: " + imp.getTitle() + "\n" + dim.toString());
        System.out.println("---------------------");

        for (int t = 1; t <= dim.batchSize; t++) {
            for (int z = 1; z <= dim.depth; z++) {

                imp.setPosition(1, z, t);
                ImageProcessor ip = imp.getProcessor();

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
     * Superimposes a grayscale mask image onto a base grayscale image. Pixels
     * with a value of 0 in the mask image will be transparent, allowing the
     * corresponding pixel from the base image to show through. Non-zero pixels
     * in the mask image will be rendered with the specified color. The method
     * handles multi-slice (stack) and hyperstack images by processing each
     * corresponding slice.
     *
     * @param maskImage The mask ImagePlus object, which should also be a
     * grayscale image. Its non-zero pixel values will determine where the color
     * is applied, and zero values will be transparent.
     * @param overlayColor The java.awt.Color to use for the non-zero pixels
     * from the mask image.
     * @return A new ImagePlus object representing the superimposed image. This
     * will be an 8-bit RGB image if successful. Returns null if input images
     * are invalid or incompatible.
     * @throws IllegalArgumentException If the input images are null, do not
     * have matching dimensions, or are not grayscale.
     */
    public ImagePlus superimposeMask(ImagePlus maskImage, Color overlayColor) {

        validateOverlay(maskImage);
        int totalSlices = imp.getStackSize();

        ImageStack outputStack = dim.getImageStack();

        for (int i = 1; i <= totalSlices; i++) {
            IJ.showProgress(i, totalSlices);
            ImageProcessor outputIp = processSlice(maskImage, overlayColor, i);
            outputStack.addSlice(imp.getStack().getSliceLabel(i), outputIp);
        }
        IJ.showProgress(1.0);

        return createFinalImagePlus(
                outputStack,
                imp.getTitle() + "_" + maskImage.getTitle() + "_Overlay"
        );
    }

    /**
     * Validates the input ImagePlus objects for null values, matching
     * dimensions, and grayscale type.
     *
     * @param imp The base ImagePlus object.
     * @param maskImage The mask ImagePlus object.
     * @throws IllegalArgumentException If validation fails.
     */
    private void validateOverlay(ImagePlus maskImage) {
        if (imp == null || maskImage == null) {
            throw new IllegalArgumentException("Input ImagePlus objects cannot be null.");
        }
//
//        if (imp.getWidth() != maskImage.getWidth())
//            throw new RuntimeException("botom layer width = " + imp.getWidth() + " and vector layer has " + maskImage.getWidth());
//
//        if (imp.getHeight()!= maskImage.getHeight())
//            throw new RuntimeException("botom layer height= " + imp.getHeight()+ " and vector layer has " + maskImage.getHeight());

        if (imp.getStackSize()!= maskImage.getStackSize())
            throw new RuntimeException("botom layer stack size = " + imp.getStackSize()+ " and vector layer has " + maskImage.getStackSize());

        if (imp.getType() == ImagePlus.COLOR_RGB || maskImage.getType() == ImagePlus.COLOR_RGB) {
            throw new IllegalArgumentException("Both input images must be grayscale (8-bit, 16-bit, or 32-bit float).");
        }
    }

    /**
     * Processes a single slice from the base and mask images, superimposing the
     * colored mask onto the base image.
     *
     * @param maskImage The mask ImagePlus object.
     * @param overlayColor The color to apply for non-zero mask pixels.
     * @param sliceIndex The 1-based index of the slice to process.
     * @return A ColorProcessor representing the processed slice.
     */
    private ImageProcessor processSlice(ImagePlus maskImage, Color overlayColor, int sliceIndex) {
        imp.setSlice(sliceIndex);
        maskImage.setSlice(sliceIndex);

        ImageProcessor baseIp = imp.getProcessor();
        ImageProcessor maskIp = maskImage.getProcessor();

        ColorProcessor outputCp = new ColorProcessor(imp.getWidth(), imp.getHeight());
        BufferedImage outputBufferedImage = (BufferedImage) outputCp.createImage();
        Graphics2D g2d = outputBufferedImage.createGraphics();

        g2d.drawImage(baseIp.createImage(), 0, 0, null);
        g2d.setColor(overlayColor);

        drawMaskPixels(g2d, maskIp);

        g2d.dispose();
        return outputCp;
    }

    /**
     * Draws the non-zero pixels from the mask ImageProcessor onto the
     * Graphics2D context with the currently set color.
     *
     * @param g2d The Graphics2D context to draw upon.
     * @param maskIp The ImageProcessor representing the mask slice.
     */
    private void drawMaskPixels(Graphics2D g2d, ImageProcessor maskIp) {
        for (int y = 0; y < dim.height; y++) {
            for (int x = 0; x < dim.width; x++) {
                if (maskIp.getPixelValue(x, y) != 0.0f) {
                    g2d.fillRect(x, y, 1, 1);
                }
            }
        }
    }

    /**
     * Creates the final ImagePlus object from the processed stack and sets its
     * hyperstack dimensions if the original base image was a hyperstack.
     *
     * @param outputStack The ImageStack containing the processed slices.
     * @param title The title for the new ImagePlus.
     * @param imp The original base ImagePlus to copy hyperstack dimensions
     * from.
     * @return The newly created ImagePlus object.
     */
    private ImagePlus createFinalImagePlus(ImageStack outputStack, String title) {
        ImagePlus finalImp = new ImagePlus(title, outputStack);

        if (imp.isHyperStack()) {
            finalImp.setDimensions(
                    imp.getNChannels(),
                    imp.getNSlices(),
                    imp.getNFrames()
            );
        }
        return finalImp;
    }
}