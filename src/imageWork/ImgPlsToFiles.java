package imageWork;

import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import java.io.File;

/**
 *
 * @author E. Dov Neimand
 */
public class ImgPlsToFiles {

    /**
     * Converts a float32 grayscale ImagePlus to a normalized 8-bit grayscale ImagePlus.
     * This is useful for saving or viewing scientific float images in standard image viewers.
     * This method correctly handles multi-slice/multi-frame ImagePlus objects by processing
     * each slice's processor individually.
     *
     * @param floatImage The original ImagePlus, assumed to contain FloatProcessors.
     * Its display range should be set to define the scaling.
     * @return A new ImagePlus with pixel values scaled to the 8-bit range [0, 255].
     */
    public static ImagePlus normalizeFloatImageTo8Bit(ImagePlus floatImage) {
        // Create a new ImageStack for the 8-bit processors
        ImageStack byteStack = new ImageStack(floatImage.getWidth(), floatImage.getHeight());

        // Get the total number of slices in the original image (linearized)
        int totalSlices = floatImage.getStackSize();

        // Iterate through each slice of the original float ImagePlus
        for (int i = 1; i <= totalSlices; i++) {
            floatImage.setSlice(i); // Activate the current slice
            ImageProcessor currentFloatProcessor = floatImage.getProcessor(); // Get its processor

            if (!(currentFloatProcessor instanceof FloatProcessor)) {
                // Handle cases where a slice might not be a FloatProcessor (though unlikely given your context)
                throw new IllegalArgumentException("Slice " + i + " is not a FloatProcessor.");
            }

            FloatProcessor fp = (FloatProcessor) currentFloatProcessor;
            
            //fp.setMinAndMax(0, Math.PI);

            // Crucial: Set the min/max on the FloatProcessor itself based on the ImagePlus's display range.
            // This ensures consistent scaling across all slices.
            // Assuming the display range is set on the 'floatImage' ImagePlus.
            double min = floatImage.getDisplayRangeMin();
            double max = floatImage.getDisplayRangeMax();

            // ImageJ's `convertToByteProcessor()` method on ImageProcessor
            // uses the min/max set on the processor (or its ImagePlus).
            fp.setMinAndMax(min, max);
            ByteProcessor bp = fp.convertToByteProcessor(); // This is the correct method call

            // Add the converted ByteProcessor to the new stack
            byteStack.addSlice(floatImage.getStack().getSliceLabel(i), bp);
        }

        // Create a new ImagePlus from the 8-bit stack
        ImagePlus normalizedImp = new ImagePlus(floatImage.getTitle() + "_8bit", byteStack);

        // If the original was a hyperstack, convert the new one as well
        if (floatImage.isHyperStack()) {
            normalizedImp.setDimensions(
                floatImage.getNChannels(),
                floatImage.getNSlices(),
                floatImage.getNFrames()
            );
            // HyperStackConverter.toHyperStack is also an option if setDimensions is not enough.
            // normalizedImp = HyperStackConverter.toHyperStack(normalizedImp, floatImage.getNChannels(), floatImage.getNSlices(), floatImage.getNFrames());
        }
        
        return normalizedImp;
    }


    /**
     * Saves each slice (channel, Z-slice, and T-frame) of an ImagePlus
     * to a separate JPEG file. The saved files will be named based on the original filename,
     * channel number, frame number, and slice number.
     * The images will be saved in the specified directory.
     *
     * @param imp The ImagePlus object to process. This should be the original
     * Float-based ImagePlus that has its display range set.
     * @param saveTo The folder the file should be saved in.
     */
    public static void saveSlices(ImagePlus imp, String saveTo) {
        printImageValues(imp);
        
        
        // First, normalize the entire ImagePlus to 8-bit.
        // This returns a *new* ImagePlus with ByteProcessors in its stack.
        ImagePlus imp8bit = normalizeFloatImageTo8Bit(imp);

        // Get the base file name from the 8-bit ImagePlus
        String baseFileName = imp8bit.getTitle();

        // Ensure the save directory exists
        File saveDir = new File(saveTo);
        if (!saveDir.exists()) {
            if (!saveDir.mkdirs()) {
                System.err.println("Failed to create directory: " + saveTo);
                return;
            }
        }

        int numChannels = imp8bit.getNChannels();
        int numSlicesZ = imp8bit.getNSlices();
        int numFrames = imp8bit.getNFrames();

        // Iterate through all dimensions (C, Z, T)
        for (int frame = 1; frame <= numFrames; frame++) {
            for (int sliceZ = 1; sliceZ <= numSlicesZ; sliceZ++) {
                for (int channel = 1; channel <= numChannels; channel++) {

                    // Calculate the 1-based linear stack index for setSlice()
                    // This uses ImageJ's internal calculation for hyperstack indexing
                    int linearStackIndex = imp8bit.getStackIndex(channel, sliceZ, frame);

                    // Set the current slice in the ImagePlus object.
                    // This makes the corresponding ImageProcessor active.
                    imp8bit.setSlice(linearStackIndex);

                    // Get the ImageProcessor for the active slice.
                    // This will be a ByteProcessor because we normalized imp8bit earlier.
                    ImageProcessor currentProcessor = imp8bit.getProcessor();

                    // Create a temporary ImagePlus object containing just this single 2D slice
                    ImagePlus sliceToSave = new ImagePlus(
                        baseFileName + "_C" + channel + "_F" + frame + "_Z" + sliceZ,
                        currentProcessor
                    );

                    FileSaver fs = new FileSaver(sliceToSave);

                    // Construct the full path and filename with .jpeg extension
                    String specificFileName = sliceToSave.getTitle() + ".jpeg";
                    String fullPath = new File(saveTo, specificFileName).getAbsolutePath();

                    // Save as JPEG
                    try {
                        if (!fs.saveAsJpeg(fullPath)) {
                            System.err.println("Failed to save JPEG image: " + fullPath);
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
     * The values are printed as floats.
     * For multi-slice images, it iterates through each slice (z-dimension).
     * For multi-channel images, it iterates through each channel.
     * For multi-frame images, it iterates through each frame (t-dimension).
     *
     * @param imp The ImagePlus object to print.
     */
    public static void printImageValues(ImagePlus imp) {
        if (imp == null) {
            System.out.println("Error: ImagePlus object is null.");
            return;
        }

        int width = imp.getWidth();
        int height = imp.getHeight();
        int nChannels = imp.getNChannels();
        int nSlices = imp.getNSlices();
        int nFrames = imp.getNFrames();
        int imageType = imp.getType();

        System.out.println("--- Image Details ---");
        System.out.println("Title: " + imp.getTitle());
        System.out.println("Width: " + width);
        System.out.println("Height: " + height);
        System.out.println("Channels: " + nChannels);
        System.out.println("Slices (Z): " + nSlices);
        System.out.println("Frames (T): " + nFrames);        
        System.out.println("---------------------");

        // Loop through frames (t-dimension)
        for (int t = 1; t <= nFrames; t++) {
            // Loop through slices (z-dimension)
            for (int z = 1; z <= nSlices; z++) {
                // Loop through channels
                for (int c = 1; c <= nChannels; c++) {

                    // Set the current position in the stack
                    imp.setPosition(c, z, t);
                    ImageProcessor ip = imp.getProcessor();

                    System.out.println("\n--- Values for Channel " + c + ", Slice " + z + ", Frame " + t + " ---");
                    for (int y = 0; y < height; y++) {
                        for (int x = 0; x < width; x++) {
                            // Get the pixel value as a float
                            float value = ip.getf(x, y);
                            System.out.printf("%8.2f ", value); // Format to 2 decimal places for readability
                        }
                        System.out.println(); // New line after each row
                    }
                }
            }
        }
        System.out.println("\n--- End of Image Values ---");
    }
    
}