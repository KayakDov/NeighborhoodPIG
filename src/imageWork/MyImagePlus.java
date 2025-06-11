package imageWork;

import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.FileSaver;
import ij.plugin.HyperStackConverter;
import ij.process.ByteProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.awt.Color;
import java.awt.Rectangle;
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
        setDimensions(1, depth, imp.size()/depth);        
        dim = new Dimensions(imp, depth);

    }

    /**
     * Constructs a MyImagePlus by taking an existing ImagePlus. This
     * constructor copies references to the ImageStack and its ImageProcessors
     * to avoid memory duplication, and copies all relevant metadata. This
     * effectively "upgrades" an existing ImagePlus to a MyImagePlus without
     * creating new pixel data.
     *
     * @param imp The ImagePlus to copy from.
     */
    public MyImagePlus(ImagePlus imp) {
        // Call the superclass constructor using the original ImagePlus's stack reference.
        // This is efficient as it doesn't duplicate the pixel data.
        super(imp.getTitle(), imp.getImageStack());

        // Copy all relevant metadata from the original ImagePlus
        this.setCalibration(imp.getCalibration()); // Copies spatial calibration
        this.setLut(imp.getProcessor().getLut()); // Copies the LUT (if applicable to the first slice's processor)
        this.setOverlay(imp.getOverlay()); // Copies the overlay (if any)
        this.setRoi(imp.getRoi()); // Copies the ROI (if any)
        this.setDimensions(1, imp.getNSlices(), imp.getNFrames()); // Copies C, Z, T dimensions
        this.setDisplayRange(imp.getDisplayRangeMin(), imp.getDisplayRangeMax()); // Copies display range
        this.setActivated(); // Makes it behave like an active image

        this.dim = new Dimensions(imp.getImageStack(), imp.getNSlices());
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
        String fileNameOnly = new File(getTitle()).getName();
        String baseFileName = fileNameOnly.startsWith("_") ? fileNameOnly.substring(1) : fileNameOnly;

        ImagePlus baseImpForSaving = prepareImageForSaving();

        ImageStack flattenedStack = new ImageStack(baseImpForSaving.getWidth(), baseImpForSaving.getHeight());

        Overlay originalOverlay = this.getOverlay();

        int numSlicesTotal = baseImpForSaving.getStackSize(); // Total linear slices (C*Z*T)
        int numChannels = baseImpForSaving.getNChannels();
        int numSlicesZ = baseImpForSaving.getNSlices();
        int numFrames = baseImpForSaving.getNFrames();

        for (int frame = 1; frame <= numFrames; frame++) {
            for (int sliceZ = 1; sliceZ <= numSlicesZ; sliceZ++) {
                for (int channel = 1; channel <= numChannels; channel++) {
                    int linearStackIndex = baseImpForSaving.getStackIndex(channel, sliceZ, frame);
                    baseImpForSaving.setSlice(linearStackIndex);

                    ImageProcessor currentProcessor = baseImpForSaving.getProcessor();
                    ImagePlus sliceToProcess = new ImagePlus("", currentProcessor.duplicate()); // Create a temporary ImagePlus for the current slice

                    if (originalOverlay != null) {
                        Overlay sliceOverlay = new Overlay();
                        for (int i = 0; i < originalOverlay.size(); i++) {
                            Roi roi = originalOverlay.get(i);
                            if (roi.getPosition() == 0 || roi.getPosition() == linearStackIndex) { // Position 0 means all slices, otherwise specific slice
                                sliceOverlay.add(roi);
                            }
                        }
                        sliceToProcess.setOverlay(sliceOverlay);
                        sliceToProcess = sliceToProcess.flatten(); 
                    }
                    
                    flattenedStack.addSlice(baseImpForSaving.getStack().getSliceLabel(linearStackIndex), sliceToProcess.getProcessor());
                }
            }
        }

        ImagePlus finalImpToSave = new ImagePlus(baseImpForSaving.getTitle() + "_flattened", flattenedStack);
        finalImpToSave.setDimensions(numChannels, numSlicesZ, numFrames);
        finalImpToSave.setCalibration(baseImpForSaving.getCalibration());

        for (int frame = 1; frame <= numFrames; frame++) {
            for (int sliceZ = 1; sliceZ <= numSlicesZ; sliceZ++) {
                for (int channel = 1; channel <= numChannels; channel++) {
                    int linearStackIndex = finalImpToSave.getStackIndex(channel, sliceZ, frame);
                    finalImpToSave.setSlice(linearStackIndex);

                    ImageProcessor currentFlattenedProcessor = finalImpToSave.getProcessor();
                    
                    ImagePlus sliceToSave = new ImagePlus(
                            baseFileName + "_C" + channel + "_F" + frame + "_Z" + sliceZ,
                            currentFlattenedProcessor
                    );

                    FileSaver fs = new FileSaver(sliceToSave);
                    String specificFileName = sliceToSave.getTitle() + ".png";
                    File outputFile = new File(saveTo, specificFileName);

                    File parentDir = outputFile.getParentFile();
                    if (parentDir != null && !parentDir.exists()) {
                        if (!parentDir.mkdirs()) {
                            System.err.println("Failed to create directory: " + parentDir.getAbsolutePath());
                            continue;
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
    public MyImagePlus overlay(ImageStack overlayStack, Color color) {
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

    /**
     * Returns a new ImagePlus that uses the same processors as a subset of
     * frames from this ImagePlus.
     *
     * @param start The 0-based index of the first frame to include (inclusive).
     * @param numFrames The number of frames to include in the subset.
     * @return A new ImagePlus containing the specified subset of frames,
     * sharing the underlying ImageProcessor objects, or null if input is
     * invalid.
     */
    public MyImagePlus subset(int start, int numFrames) {

        numFrames = Math.min(numFrames, getNFrames() - start);

        int end = start + numFrames;

        ImageStack subsetStack = new ImageStack(dim.width, dim.height);

        for (int frame = start; frame < end; frame++)
            for(int slice = 0; slice < dim.depth; slice++)
                subsetStack.addSlice(
                        getStack().getSliceLabel(frame*dim.depth + slice + 1), 
                        getStack().getProcessor(frame*dim.depth + slice + 1)
                );

        MyImagePlus newImp = new MyImagePlus(
                getTitle() + "_subset_F" + (start + 1) + "_to_F" + end,
                subsetStack,
                dim.depth
        );

        return newImp;
    }

    /**
     * Crops this image down to the new height and width.
     *
     * @param width The new width of this image.
     * @param height The new height of this image.
     * @return this image.
     */
    public MyImagePlus crop(int height, int width) {
        dim = new Dimensions(height, width, dim.depth, dim.batchSize);
        setRoi(new Roi(new Rectangle(0, 0, width, height)));
        setStack(crop("stack").getImageStack());
        deleteRoi();
        return this;
    }

    /**
     * The dimensions.
     *
     * @return The dimensions.
     */
    public Dimensions dim() {
        return dim;
    }

}
