package imageWork;

import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ImageProcessor;
import ij.process.ColorProcessor;
import ij.process.FloatProcessor;
import ij.process.ByteProcessor;
import ij.process.ShortProcessor;
import ij.LookUpTable;
import ij.plugin.LutLoader;
import ij.process.LUT;
import java.awt.Color;
import java.awt.image.IndexColorModel;

/**
 * A utility class for ImageJ/Fiji to apply a spectrum Look-Up Table (LUT) to
 * grayscale images, specifically rendering NaN (Not a Number) values as black.
 * This is particularly useful for 32-bit float images that may contain NaN.
 */
public class LUTWithNaN {//TODO: Not yet tested.

    private Dimensions dim;

    /**
     * Helper class to encapsulate min/max values and a flag indicating if valid
     * non-NaN data was found.
     */
    private class MinMaxResult {

        double minVal;
        double maxVal;

        /**
         * Check if ch is the new min or max, and if so, set it as such.
         *
         * @param ch A value that might be the new min or max.
         */
        private void chalenge(double ch) {
            if (Double.isNaN(ch)) return;
            minVal = Math.min(minVal, ch);
            maxVal = Math.max(maxVal, ch);
        }

        /**
         * Finds the minimum and maximum pixel values in the image plus.
         *
         * @param ip The image plus for which the min and max pixel values are
         * desired.
         */
        MinMaxResult(ImagePlus ip) {
            for (int i = 1; i <= ip.getStackSize(); i++) {
                ip.setSlice(i);
                FloatProcessor fp = (FloatProcessor) ip.getProcessor();
                for (int y = 0; y < dim.height; y++)
                    for (int x = 0; x < dim.width; x++)
                        chalenge(fp.getf(x, y));
            }
        }

        /**
         * The range of the images.
         *
         * @return The range of the images.
         */
        double range() {
            return maxVal - minVal;
        }
    }

    private byte[] r, g, b;

    /**
     * Sets up the LUT
     */
    public LUTWithNaN() {
        IndexColorModel spectrumICM = LutLoader.getLut("Spectrum");
        r = new byte[256];
        g = new byte[256];
        b = new byte[256];
        spectrumICM.getReds(r);
        spectrumICM.getGreens(g);
        spectrumICM.getBlues(b);
    }

    /**
     * Calculates the RGB color from a pixel value using the provided LUT
     * arrays.
     *
     * @param pixelValue The value of the pixel.
     * @param minVal The minimum valid pixel value.
     * @param range The range of valid pixel values.
     * @return The calculated RGB color.
     */
    public int getColor(float pixelValue, double minVal, double range) {
        if (Double.isNaN(pixelValue)) return 0;
        int lutIndex = (int) (((pixelValue - minVal) / range) * 255.0);
        lutIndex = Math.max(0, Math.min(255, lutIndex)); // Clamp index
        return new Color(r[lutIndex] & 0xFF, g[lutIndex] & 0xFF, b[lutIndex] & 0xFF).getRGB();
    }

    /**
     * Iterates through each pixel of the image and applies the spectrum LUT,
     * handling NaN values specifically for float images.
     *
     * @param fp The input float processor.
     * @param mmr The range data.
     * @return
     */
    private ColorProcessor toColor(FloatProcessor fp, MinMaxResult mmr) {
        ColorProcessor cp = dim.getColorProcessor((float) mmr.minVal, (float) mmr.range());
        for (int y = 0; y < dim.height; y++)
            for (int x = 0; x < dim.width; x++)
                cp.set(x, y, getColor(fp.getf(x, y), mmr.minVal, mmr.range()));
        return cp;
    }

    /**
     * Creates a colored image from the gray scale.
     *
     * @param ip A gray scale image.
     * @return A color iamge.
     */
    public ImagePlus applyLutToImage(ImagePlus ip) {
        MinMaxResult mmr = new MinMaxResult(ip);
        MyImageStack to = dim.getImageStack();
        for (int i = 0; i < ip.getStackSize(); i++) {
            ip.setSlice(i);
            to.addSlice(toColor((FloatProcessor) ip.getProcessor(), mmr));
        }
        return new MyImagePlus(ip.getTitle() + " LUT", to, ip.getNSlices());
    }

    /**
     * Applies a spectrum Look-Up Table (LUT) to a grayscale ImagePlus.
     *
     * <p>
     * For 32-bit float images, this method identifies and renders NaN values as
     * pure black (RGB: 0,0,0). Non-NaN values are mapped to the spectrum based
     * on their normalized range within the image's valid data. For 8-bit and
     * 16-bit images, a standard spectrum LUT is applied.</p>
     *
     * @param imp The input grayscale ImagePlus (8-bit, 16-bit, or 32-bit
     * float).
     * @return A new ImagePlus object with the spectrum LUT applied. NaN values
     * in float images will be black. Returns {@code null} if the input image is
     * null or not a supported grayscale type.
     */
    public ImagePlus applySpectrumLUT(ImagePlus imp) {
        ImageProcessor ip = validateInput(imp);
        if (ip == null) return null;

        dim = new Dimensions(imp);
        ColorProcessor cp = new ColorProcessor(dim.width, dim.height);

        MinMaxResult minMaxResult = new MinMaxResult(imp);

        ImagePlus resultImp = applyLutToImage(imp);
        resultImp.copyAttributes(imp);
        return resultImp;
    }

    /**
     * Validates the input ImagePlus, ensuring it's not null and is a supported
     * grayscale type (8-bit, 16-bit, or 32-bit float).
     *
     * @param imp The ImagePlus to validate.
     * @return The ImageProcessor of the input image if valid, otherwise
     * {@code null}.
     */
    private ImageProcessor validateInput(ImagePlus imp) {
        if (imp == null) {
            System.err.println("Error: Input ImagePlus is null. Cannot apply LUT.");
            return null;
        }
        ImageProcessor ip = imp.getProcessor();
        if (!(ip instanceof ByteProcessor || ip instanceof ShortProcessor || ip instanceof FloatProcessor)) {
            System.err.println("Error: Input image must be grayscale (8-bit, 16-bit, or 32-bit float). "
                    + "Current type: " + ip.getClass().getSimpleName());
            return null;
        }
        return ip;
    }

}
