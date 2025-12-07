package imageWork;

import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.ImageRoi;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.io.FileSaver;
import ij.plugin.AVI_Reader;
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
     * @param depth The depth of the image.
     */
    public MyImagePlus(String title, ImageStack imp, int depth) {
        super(title, imp);
        setDimensions(1, depth, imp.size() / depth);
        dim = new Dimensions(imp, depth);
//        dim.setToHyperStack(this);  I don't think this is neccessary after setDimensions.
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
        
        dim = new Dimensions(imp.getImageStack(), imp.getNSlices());
        dim.setToHyperStack(this);
    }

    /**
     * Converts a float32 grayscale ImagePlus to a normalized 8-bit grayscale
     * ImagePlus.This method correctly handles multi -slice /multi -frame
     * ImagePlus objects by processing each slice's processor individually.
     *
     * @return A new ImagePlus with pixel values scaled to the 8-bit range[0,
     * 255].
     */
    public MyImagePlus normalizeFloatImageTo8Bit() {

        System.out.println("imageWork.MyImagePlus.normalizeFloatImageTo8Bit() normalizing image");

        // Use the display range from the original MyImagePlus for normalization.
        // This is what ImageJ is already using to display the image correctly.
        double displayMin = this.getDisplayRangeMin();
        double displayMax = this.getDisplayRangeMax();

        // Handle cases where the display range might be degenerate (e.g., solid color)
        // or if min == max, adjust to prevent division by zero in scaling.
        if (displayMax == displayMin) {
            if (displayMin == 0.0) {
                displayMax = 1.0; // Ensure a valid range for conversion, 0 will map to 0
            } else {
                displayMin = 0.0; // Set min to 0 to ensure the constant non-zero value maps to max (255)
            }
        }

        ImageStack byteStack = new ImageStack(getWidth(), getHeight());
        int totalSlices = getStackSize();

        for (int i = 1; i <= totalSlices; i++) {
            setSlice(i);
            ImageProcessor currentFloatProcessor = getProcessor(); // Get its processor

            if (!(currentFloatProcessor instanceof FloatProcessor))
                throw new IllegalArgumentException("Slice " + i + " is not a FloatProcessor.");

            FloatProcessor fp = (FloatProcessor) currentFloatProcessor.duplicate(); // Duplicate to avoid modifying original or impacting other references

            // Apply the image's current display min/max for conversion
            fp.setMinAndMax(displayMin, displayMax);

            ByteProcessor bp = fp.convertToByteProcessor(); // This now uses the specified min/max for scaling

            byteStack.addSlice(getStack().getSliceLabel(i), bp);
        }

        // Create the new ImagePlus with the 8-bit stack
        MyImagePlus normalizedImp = new MyImagePlus(getTitle() + "_8bit", byteStack, dim.depth);

        // Explicitly set display range for the new 8-bit image to ensure correct visualization and saving
        normalizedImp.setDisplayRange(0, 255);
        return normalizedImp;
    }

    /**
     * Prepares the image for saving. If the image is a 32-bit float, it is
     * normalized to 8-bit. Otherwise, a copy of the original image is created.
     * This makes the saveSlices method more robust for different image types.
     *
     * @param toTiff true if the image will be saved as a tiff, false otherwise.
     * @return An ImagePlus ready for saving.
     */
    private MyImagePlus prepareImageForSaving(boolean toTiff) {

        return getType() == ImagePlus.GRAY32 && !toTiff ? normalizeFloatImageTo8Bit() : this;
    }

    /**
     * Saves each slice of this image to individual PNG files. Normalizes float
     * images, applies overlays, flattens, and names output files based on Z and
     * T.
     *
     * @param saveTo Directory to save the PNGs to.
     * @param toTiff True if the image should be saved to a tiff, false
     * otherwise.
     */
    public void saveSlices(String saveTo, boolean toTiff) {

        ImagePlus prepared = prepareImageForSaving(toTiff);

        ImagePlus flattened = buildFlattenedImage(prepared);
        writeSlicesToDisk(flattened, saveTo, toTiff);
    }

    /**
     * Flattens the image with overlays and returns a new ImagePlus.
     *
     * @param imp The image to flatten.
     * @return A new flattened ImagePlus with same dimensions.
     */
    private ImagePlus buildFlattenedImage(ImagePlus imp) {
        int width = imp.getWidth();
        int height = imp.getHeight();
        int slices = imp.getNSlices();
        int frames = imp.getNFrames();

        ImageStack stack = new ImageStack(width, height);
        Overlay overlay = getOverlay(); // Get overlay from *this* MyImagePlus

        for (int t = 1; t <= frames; t++) {
            for (int z = 1; z <= slices; z++) {
                int index = imp.getStackIndex(1, z, t);
                imp.setSlice(index);

                ImagePlus slice = new ImagePlus("", imp.getProcessor().duplicate());

                if (overlay != null) {
                    Overlay sliceOverlay = extractOverlayForSlice(overlay, index);
                    if (sliceOverlay != null) {
                        slice.setOverlay(sliceOverlay);
                        slice = slice.flatten(); // Flatten this slice with its overlay
                    }
                }
                stack.addSlice(imp.getStack().getSliceLabel(index), slice.getProcessor());
            }
        }

        ImagePlus result = new ImagePlus(imp.getTitle() + "_flattened", stack);
        result.setDimensions(1, slices, frames);
        result.setCalibration(imp.getCalibration());
        return result;
    }

    /**
     * Extracts the overlay relevant for a specific slice index.
     *
     * @param original The original overlay.
     * @param targetIndex The index in the stack.
     * @return A new Overlay or null if nothing applies.
     */
    private Overlay extractOverlayForSlice(Overlay original, int targetIndex) {
        Overlay result = new Overlay();
        for (int i = 0; i < original.size(); i++) {
            Roi roi = original.get(i);
            int pos = roi.getPosition();
            if (pos == 0 || pos == targetIndex)
                result.add(roi);
        }
        return result.size() > 0 ? result : null;
    }

    /**
     * Writes all slices of the given ImagePlus to disk as PNG files.
     *
     * @param imp Image with flattened slices.
     * @param saveTo Directory to save images.
     * @param toTiff True to save to a tiff file, false for png.
     */
    private void writeSlicesToDisk(ImagePlus imp, String saveTo, boolean toTiff) {
        int slices = imp.getNSlices();
        int frames = imp.getNFrames();
        String baseFileName = new File(getTitle()).getName().replaceFirst("^_", "");

        System.out.println("Saving to: " + saveTo);
        for (int t = 1; t <= frames; t++) {
            for (int z = 1; z <= slices; z++) {
                int index = imp.getStackIndex(1, z, t);
                imp.setSlice(index);

                ImagePlus sliceToSave = new ImagePlus(
                        baseFileName + "_F" + t + "_Z" + z,
                        imp.getProcessor().duplicate() // Duplicate the processor to ensure isolated saving
                );

                // If it's an 8-bit image, ensure its display range is set for proper saving
                if (sliceToSave.getProcessor() instanceof ByteProcessor || sliceToSave.getProcessor() instanceof ColorProcessor)
                    sliceToSave.setDisplayRange(0, 255);

                File outFile = new File(saveTo, sliceToSave.getTitle() + (toTiff ? ".tiff" : ".png"));
                ensureDir(outFile.getParentFile());

                FileSaver fs = new FileSaver(sliceToSave);

                if (toTiff || !fs.saveAsPng(outFile.getAbsolutePath()) && (!toTiff || !fs.saveAsTiff(outFile.getAbsolutePath())))
                    System.err.println("Failed to save: " + outFile.getAbsolutePath());
            }
        }
    }

    /**
     * Ensures the directory exists, creating it if necessary.
     *
     * @param dir The directory to check/create.
     */
    private void ensureDir(File dir) {
        if (dir != null && !dir.exists() && !dir.mkdirs()) {
            System.err.println("Could not create directory: " + dir.getAbsolutePath());
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

        ImageRoi roi;
        
        for (int z = 1; z <= overlayStack.getSize(); z++) {
            ImageProcessor processor = overlayStack.getProcessor(z);

            if (!(processor instanceof ColorProcessor)) {
                // Create a color image from binary mask
                ColorProcessor colorProcessor = new ColorProcessor(width, height);
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++) {

                        if (processor.get(x, y) == 255)
                            colorProcessor.set(x, y, color.getRGB());
                        else colorProcessor.set(x, y, 0);
                    }
                roi = new ImageRoi(0, 0, colorProcessor);
            } else roi = new ImageRoi(0, 0, processor);
            
            roi.setZeroTransparent(true);
            roi.setPosition(z);

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
            for (int slice = 0; slice < dim.depth; slice++) {
                int index = getStackIndex(1, slice + 1, frame + 1);
                subsetStack.addSlice(
                        getStack().getSliceLabel(index),
                        getStack().getProcessor(index)
                );
            }

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
     * @param depth The new depth.
     * @return this image.
     */
    public MyImagePlus crop(int height, int width, int depth) {
    
        setRoi(new Roi(new Rectangle(0, 0, width, height)));

        for (int t = 0; t < dim.batchSize; t++)
            for (int z = depth; z < dim.depth; z++)
                getStack().deleteSlice((t + 1) * depth + 1);

        setStack(crop("stack").getImageStack());

        deleteRoi();
        
        setDimensions(1, depth, dim.batchSize);
        dim = new Dimensions(height, width, depth, dim.batchSize);
        
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
