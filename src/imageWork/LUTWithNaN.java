//package imageWork;
//
//import fijiPlugin.Dimensions;
//import ij.ImagePlus;
//import ij.process.ImageProcessor;
//import ij.process.ColorProcessor;
//import ij.process.FloatProcessor;
//import ij.process.ByteProcessor;
//import ij.process.ShortProcessor;
//import ij.LookUpTable;
//import java.awt.Color;
//
///**
// * A utility class for ImageJ/Fiji to apply a spectrum Look-Up Table (LUT) to
// * grayscale images, specifically rendering NaN (Not a Number) values as black.
// * This is particularly useful for 32-bit float images that may contain NaN.
// */
//public class LUTWithNaN {
//
//    private Dimensions dim;
//
//    /**
//     * Helper class to encapsulate min/max values and a flag indicating if valid
//     * non-NaN data was found.
//     */
//    private class MinMaxResult {
//
//        double minVal;
//        double maxVal;
//        boolean hasValidNonNaNRange;
//
//        /**
//         * Check if ch is the new min or max, and if so, set it as such.
//         *
//         * @param ch A value that might be the new min or max.
//         */
//        private void chalenge(double ch) {
//            minVal = Math.min(minVal, ch);
//            maxVal = Math.max(maxVal, ch);
//        }
//
//        MinMaxResult(ImagePlus ip) {
//            for (int i = 1; i <= ip.getStackSize(); i++) {
//                ip.setSlice(i);
//                FloatProcessor fp = (FloatProcessor) ip.getProcessor();
//                for (int y = 0; y < dim.height; y++) {
//                    for (int x = 0; x < dim.width; x++) {
//                        float value = fp.getf(x, y);
//                        if (!Float.isNaN(value)) {
//                            chalenge(value);
//                            hasValidNonNaNRange = true;
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    /**
//     * Applies a spectrum Look-Up Table (LUT) to a grayscale ImagePlus.
//     *
//     * <p>
//     * For 32-bit float images, this method identifies and renders NaN values as
//     * pure black (RGB: 0,0,0). Non-NaN values are mapped to the spectrum based
//     * on their normalized range within the image's valid data. For 8-bit and
//     * 16-bit images, a standard spectrum LUT is applied.</p>
//     *
//     * @param imp The input grayscale ImagePlus (8-bit, 16-bit, or 32-bit
//     * float).
//     * @return A new ImagePlus object with the spectrum LUT applied. NaN values
//     * in float images will be black. Returns {@code null} if the input image is
//     * null or not a supported grayscale type.
//     */
//    public ImagePlus applySpectrumLUT(ImagePlus imp) {
//        ImageProcessor ip = validateInput(imp);
//        if (ip == null) return null;
//
//        dim = new Dimensions(imp);
//        ColorProcessor cp = new ColorProcessor(dim.width, dim.height);
//
//        MinMaxResult minMaxResult = findMinMax(ip, width, height);
//
//        if (handleEdgeCases(imp, ip, cp, minMaxResult)) {
//            return new ImagePlus(imp.getTitle() + " - Spectrum (NaN Black)", cp);
//        }
//
//        LookUpTable spectrumLUT = LookUpTable.createLut(LookUpTable.SPECTRUM);
//        byte[] r = spectrumLUT.getReds();
//        byte[] g = spectrumLUT.getGreens();
//        byte[] b = spectrumLUT.getBlues();
//
//        double range = minMaxResult.maxVal - minMaxResult.minVal;
//        if (range == 0) {
//            range = 1.0;
//        }
//
//        applyLutToImage(ip, cp, minMaxResult.minVal, range, r, g, b, ip instanceof FloatProcessor);
//
//        ImagePlus resultImp = new ImagePlus(imp.getTitle() + " - Spectrum (NaN Black)", cp);
//        resultImp.copyAttributes(imp);
//        return resultImp;
//    }
//
//    /**
//     * Validates the input ImagePlus, ensuring it's not null and is a supported
//     * grayscale type (8-bit, 16-bit, or 32-bit float).
//     *
//     * @param imp The ImagePlus to validate.
//     * @return The ImageProcessor of the input image if valid, otherwise
//     * {@code null}.
//     */
//    private ImageProcessor validateInput(ImagePlus imp) {
//        if (imp == null) {
//            System.err.println("Error: Input ImagePlus is null. Cannot apply LUT.");
//            return null;
//        }
//        ImageProcessor ip = imp.getProcessor();
//        if (!(ip instanceof ByteProcessor || ip instanceof ShortProcessor || ip instanceof FloatProcessor)) {
//            System.err.println("Error: Input image must be grayscale (8-bit, 16-bit, or 32-bit float). "
//                    + "Current type: " + ip.getClass().getSimpleName());
//            return null;
//        }
//        return ip;
//    }
//
//    /**
//     * Handles edge cases where the image contains only NaN values or all valid
//     * pixels have the same value. Fills the ColorProcessor accordingly.
//     *
//     * @param imp The original ImagePlus (for title).
//     * @param ip The ImageProcessor.
//     * @param cp The ColorProcessor to modify.
//     * @param minMaxResult The MinMaxResult containing min/max values and valid
//     * range flag.
//     * @return True if an edge case was handled and the image is ready to be
//     * returned, false otherwise.
//     */
//    private boolean handleEdgeCases(ImagePlus imp, ImageProcessor ip, ColorProcessor cp, MinMaxResult minMaxResult) {
//        if (!minMaxResult.hasValidNonNaNRange) {
//            System.out.println("Info: Image contains only NaN values or is empty. Filling with black.");
//            cp.setRGB(0, 0, imp.getWidth(), imp.getHeight(), 0x000000);
//            return true;
//        } else if (minMaxResult.maxVal == minMaxResult.minVal) {
//            System.out.println("Info: All non-NaN pixel values are identical. Filling with a neutral grey.");
//            int neutralColor = new Color(128, 128, 128).getRGB();
//            fillFlatImage(ip, cp, neutralColor, imp.getWidth(), imp.getHeight(), ip instanceof FloatProcessor);
//            return true;
//        }
//        return false;
//    }
//
//    /**
//     * Fills the ColorProcessor for a flat image (all valid pixels have same
//     * value), applying neutral color to valid pixels and black to NaNs.
//     *
//     * @param ip The ImageProcessor.
//     * @param cp The ColorProcessor to fill.
//     * @param neutralColor The RGB color for valid pixels.
//     * @param width The width of the image.
//     * @param height The height of the image.
//     * @param hasFloatProcessor True if the image is a FloatProcessor.
//     */
//    private void fillFlatImage(ImageProcessor ip, ColorProcessor cp, int neutralColor, int width, int height, boolean hasFloatProcessor) {
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                if (hasFloatProcessor && Float.isNaN(((FloatProcessor) ip).getf(x, y))) {
//                    cp.setRGB(x, y, 0x000000); // Black for NaN
//                } else {
//                    cp.setRGB(x, y, neutralColor); // Grey for valid, flat values
//                }
//            }
//        }
//    }
//
//    /**
//     * Iterates through each pixel of the image and applies the spectrum LUT,
//     * handling NaN values specifically for float images.
//     *
//     * @param ip The input ImageProcessor.
//     * @param cp The ColorProcessor to write the results to.
//     * @param minVal The minimum valid pixel value.
//     * @param range The range of valid pixel values (maxVal - minVal).
//     * @param r Red component array of the LUT.
//     * @param g Green component array of the LUT.
//     * @param b Blue component array of the LUT.
//     * @param hasFloatProcessor True if the image is a FloatProcessor.
//     */
//    private void applyLutToImage(ImageProcessor ip, ColorProcessor cp, double minVal, double range, byte[] r, byte[] g, byte[] b, boolean hasFloatProcessor) {
//        int width = ip.getWidth();
//        int height = ip.getHeight();
//        for (int y = 0; y < height; y++) {
//            for (int x = 0; x < width; x++) {
//                int rgbColor = getPixelColor(ip, x, y, minVal, range, r, g, b, hasFloatProcessor);
//                cp.setRGB(x, y, rgbColor);
//            }
//        }
//    }
//
//    /**
//     * Calculates the RGB color for a single pixel based on its value and the
//     * LUT. Handles NaN values for float images.
//     *
//     * @param ip The ImageProcessor.
//     * @param x The x-coordinate of the pixel.
//     * @param y The y-coordinate of the pixel.
//     * @param minVal The minimum valid pixel value.
//     * @param range The range of valid pixel values.
//     * @param r Red component array of the LUT.
//     * @param g Green component array of the LUT.
//     * @param b Blue component array of the LUT.
//     * @param hasFloatProcessor True if the image is a FloatProcessor.
//     * @return The calculated RGB color.
//     */
//    private int getPixelColor(ImageProcessor ip, int x, int y, double minVal, double range, byte[] r, byte[] g, byte[] b, boolean hasFloatProcessor) {
//        if (hasFloatProcessor) {
//            return getFloatPixelColor((FloatProcessor) ip, x, y, minVal, range, r, g, b);
//        } else {
//            return getNonFloatPixelColor(ip, x, y, minVal, range, r, g, b);
//        }
//    }
//
//    /**
//     * Calculates the RGB color for a pixel in a FloatProcessor, handling NaN.
//     *
//     * @param fp The FloatProcessor.
//     * @param x The x-coordinate.
//     * @param y The y-coordinate.
//     * @param minVal The minimum valid pixel value.
//     * @param range The range of valid pixel values.
//     * @param r Red component array of the LUT.
//     * @param g Green component array of the LUT.
//     * @param b Blue component array of the LUT.
//     * @return The calculated RGB color.
//     */
//    private int getFloatPixelColor(FloatProcessor fp, int x, int y, double minVal, double range, byte[] r, byte[] g, byte[] b) {
//        float floatVal = fp.getf(x, y);
//        if (Float.isNaN(floatVal)) {
//            return 0x000000; // Black for NaN
//        } else {
//            return calculateLutColor(floatVal, minVal, range, r, g, b);
//        }
//    }
//
//    /**
//     * Calculates the RGB color for a pixel in a non-FloatProcessor (Byte or
//     * Short).
//     *
//     * @param ip The ImageProcessor.
//     * @param x The x-coordinate.
//     * @param y The y-coordinate.
//     * @param minVal The minimum valid pixel value.
//     * @param range The range of valid pixel values.
//     * @param r Red component array of the LUT.
//     * @param g Green component array of the LUT.
//     * @param b Blue component array of the LUT.
//     * @return The calculated RGB color.
//     */
//    private int getNonFloatPixelColor(ImageProcessor ip, int x, int y, double minVal, double range, byte[] r, byte[] g, byte[] b) {
//        double pixelValue = ip.getPixelValue(x, y);
//        return calculateLutColor(pixelValue, minVal, range, r, g, b);
//    }
//
//    /**
//     * Calculates the RGB color from a pixel value using the provided LUT
//     * arrays.
//     *
//     * @param pixelValue The value of the pixel.
//     * @param minVal The minimum valid pixel value.
//     * @param range The range of valid pixel values.
//     * @param r Red component array of the LUT.
//     * @param g Green component array of the LUT.
//     * @param b Blue component array of the LUT.
//     * @return The calculated RGB color.
//     */
//    private int calculateLutColor(double pixelValue, double minVal, double range, byte[] r, byte[] g, byte[] b) {
//        int lutIndex = (int) (((pixelValue - minVal) / range) * 255.0);
//        lutIndex = Math.max(0, Math.min(255, lutIndex)); // Clamp index
//        return new Color(r[lutIndex] & 0xFF, g[lutIndex] & 0xFF, b[lutIndex] & 0xFF).getRGB();
//    }
//}
