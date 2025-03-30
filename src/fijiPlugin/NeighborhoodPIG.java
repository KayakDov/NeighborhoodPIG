package fijiPlugin;

import FijiInput.UserInput;
import imageWork.ColorHeatMapCreator;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.Opener;
import ij.process.ImageProcessor;
import imageWork.GrayScaleHeatMapCreator;
import imageWork.HeatMapCreator;
import imageWork.VectorImg;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG extends Dimensions implements AutoCloseable {

    public final StructureTensorMatrix stm; //TODO: this should probably be made private

    public final static boolean D3 = true, D2 = false;

    private final String[] sourceFileNames;

    /**
     *
     * @param image An image.
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @param tolerance How close must a number be to 0 to be considered 0.
     */
    private NeighborhoodPIG(Handle handle, FStrideArray3d image, String[] sourceFileNames, UserInput ui) {
        super(handle, image);

        this.sourceFileNames = sourceFileNames == null ? defaultNames() : sourceFileNames;

        try (Gradient grad = new Gradient(handle, image, ui.neighborhoodSize)) {
            stm = new StructureTensorMatrix(handle, grad, ui);
        }
    }

    /**
     * Names chosen for these layers when none are provided.
     *
     * @return A set of default names for the layers.
     */
    public final String[] defaultNames() {
        String[] names = new String[depth * batchSize];
        int nameIndex = 0;
        for (int frameInd = 0; frameInd < batchSize; frameInd++) {
            for (int layerInd = 0; layerInd < depth; layerInd++) {
                names[nameIndex++] = "Frame " + frameInd + " Layer " + layerInd;
            }
        }
        return names;

    }

    /**
     * A heat map of the orientation in the xy plane.
     *
     * @param color True for a color image, false for grayscale.
     * @param useCoherence True if pixel intensity should be tied to orientation
     * confidence (coherence)
     * @return A heat map of the orientation in the xy plane.
     */
    public HeatMapCreator getAzimuthalAngles(boolean color, boolean useCoherence) {

        return color ? new ColorHeatMapCreator(handle,
                concat(sourceFileNames, " Azimuth"),
                "Azimuth Angle Heatmap",
                stm.azimuthAngle(),
                useCoherence ? stm.getCoherence() : null
        )
                : new GrayScaleHeatMapCreator(
                        concat(sourceFileNames, " Azimuth"),
                        "Azimuth Angle Heatmap",
                        handle,
                        useCoherence ? stm.getCoherence() : stm.azimuthAngle()
                );
    }

    /**
     * A heat map of the orientation in the yz plane.
     *
     * @param color True for a color image, false for grayscale.
     * @param useCoherence True if pixel intensity should be tied to orientation
     * confidence (coherence). If a grayscale image is requested, than set to
     * true to get the coherence.
     * @return A heat map of the orientation in the yz plane.
     */
    public HeatMapCreator getZenithAngles(boolean color, boolean useCoherence) {
        return color ? new ColorHeatMapCreator(
                handle,
                concat(sourceFileNames, " Zenith Angle"),
                "Zenith Angle Heatmap",
                stm.zenithAngle(),
                useCoherence ? stm.getCoherence() : null
        )
                : new GrayScaleHeatMapCreator(
                        concat(sourceFileNames, useCoherence ? " Coherence" : " Zenith Angle"),
                        useCoherence ? "Coherence" : "Zenith Angle Heatmap",
                        handle,
                        useCoherence ? stm.getCoherence() : stm.zenithAngle()
                );
    }

    /**
     * An image of all the nematic vectors
     * @param spacing The space between the vectors.
     * @param vecMag The magnitude of the vectors to be drawn.
     * @return An image of all the nematic vectors
     */
    public ImagePlus getVectorImg(int spacing, int vecMag) {
        
        return new VectorImg(this).get(
                stm.getEigen().vectors1, 
                stm.getCoherence(), 
                spacing, 
                vecMag, 
                sourceFileNames
        );
    }

    /**
     * Concatenates the concatenation onto each of the Strings provided.
     *
     * @param needsConcat Strings that will be copied and have something added
     * on to them.
     * @return An array of strings copied from the input, each one with
     * concatenation added.
     */
    private String[] concat(String[] needsConcat, String concatination) {
        return Arrays.stream(needsConcat).map(nc -> {
            if (nc == null) {
                return concatination;
            }
            int dotIndex = nc.lastIndexOf('.');
            return dotIndex == -1 ? nc + concatination : nc.substring(0, dotIndex) + concatination + nc.substring(dotIndex);
        }).toArray(String[]::new);
    }

    @Override
    public void close() {
        stm.close();
    }

    /**
     * A factory method for a neighborhood pig.
     *
     * @param handle
     * @param imp The imagePlus from which the image data is taken.
     * @param ui Input submited by the user.
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG get(Handle handle, ImagePlus imp, UserInput ui) {

//        if (imp.hasImageStack() && imp.getNSlices() == 1 && imp.getNFrames() > 1)
//            HyperStackConverter.toHyperStack(imp, 1, 1, imp.getNSlices());
        try (FStrideArray3d gpuImmage = processImages(handle, imp)) {

            return new NeighborhoodPIG(
                    handle,
                    gpuImmage,
                    imp.getImageStack().getSliceLabels(),
                    ui
            );
        }

    }

    /**
     * A factory method for a neighborhood pig.
     *
     * @param handle
     * @param folderPath All images in the folder should have the sameheight and
     * width.
     * @param ui The input from the user.
     * @param depth
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG get(Handle handle, String folderPath, int depth, UserInput ui) {
        try {

            File[] imageFiles = getImageFiles(folderPath);

            BufferedImage firstImage = ImageIO.read(imageFiles[0]);

            try (FStrideArray3d gpuImage = processImages(handle, imageFiles, firstImage.getHeight(), firstImage.getWidth(), depth)) {

                return new NeighborhoodPIG(
                        handle,
                        gpuImage,
                        Arrays.stream(imageFiles).map(File::getName).toArray(String[]::new),
                        ui
                );
            }
        } catch (IOException ex) {
            Logger.getLogger(NeighborhoodPIG.class.getName()).log(Level.SEVERE, null, ex);
            throw new RuntimeException();
        }
    }

    /**
     * A factory method for a neighborhood pig.
     *
     * @param handle
     * @param folderPath All images in the folder should have the sameheight and
     * width.
     * @param nRad The radius of a neighborhood. The neighborhood will be a
     * square and the radius is the distance from the center to the nearest
     * edge.
     * @param ui User set specifications.
     * @param depth
     * @param tolerance Close enough to 0.
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG getWithIJ(Handle handle, String folderPath, int depth, UserInput ui) {

        ImagePlus ip = imagePlus(folderPath, depth);
        try (FStrideArray3d gpuImage = processImages(handle, ip)) {

            return new NeighborhoodPIG(
                    handle,
                    gpuImage,
                    new File(folderPath).list(),
                    ui
            );
        }

    }

    /**
     * Processes a hyperstack and returns a FArray containing the processed
     * image data.
     *
     * @param handle The handle used for FArray operations.
     * @param imp The ImagePlus object representing the hyperstack.
     * @return A FArray containing the image data in column-major order for all
     * frames, slices, and channels.
     */
    public final static FStrideArray3d processImages(Handle handle, ImagePlus imp) {
        // Convert the image to grayscale if necessary
        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) {
            System.out.println("fijiPlugin.NeighborhoodPIG.processImages(): Converting image to grayscale.");
            IJ.run(imp, "32-bit", "");
        }

        int width = imp.getWidth();
        int height = imp.getHeight();
        int channels = imp.getNChannels();
        int depth = imp.getNSlices();
        int frames = imp.getNFrames();
        int imgSize = width * height;

        FStrideArray3d processedImage = new FStrideArray3d(height, width, depth, frames);

        float[] columnMajorSlice = new float[imgSize];

        // Iterate over frames, slices, and channels
        for (int frame = 1; frame <= frames; frame++) {
            for (int slice = 1; slice <= depth; slice++) {
                for (int channel = 1; channel <= channels; channel++) {
                    imp.setPosition(channel, slice, frame);
                    ImageProcessor ip = imp.getProcessor();
                    float[][] pixels = ip.getFloatArray();

                    for (int col = 0; col < width; col++) {
                        for (int row = 0; row < height; row++) {
                            columnMajorSlice[col * height + row] = pixels[col][row];
                        }
                    }

                    processedImage.getSubArray(frame - 1).getLayer(slice - 1).set(handle, columnMajorSlice);
                }
            }
        }

        return processedImage;
    }

    /**
     * Copies the raster to an array in column major order.
     *
     * @param raster The raster being written from.
     * @param writeTo The array being written to.
     */
    private static void toColMjr(Raster raster, float[] writeTo) {
        for (int col = 0; col < raster.getWidth(); col++) {
            for (int row = 0; row < raster.getHeight(); row++) {
                writeTo[col * raster.getHeight() + row] = raster.getSample(col, row, 0);
            }
        }
    }

    /**
     * Converts grayscale or RGB image files in a folder into a single
     * column-major GPU array of pixel values. RGB images are converted to
     * grayscale first.
     *
     * @param handle Context.
     * @param pics Path to the folder containing image files.
     * @param height The height of the pictures.
     * @param width The width of the pictures.
     * @param depth The depth of the image.
     *
     * @return A single column-major GPU array containing pixel values of all
     * images.
     * @throws IllegalArgumentException If no valid images are found in the
     * folder.
     */
    public final static FStrideArray3d processImages(Handle handle, File[] pics, int height, int width, int depth) {

        FStrideArray3d pixelsGPU = new FStrideArray3d(height, width, depth, pics.length / depth);
        int imgSize = width * height;
        float[] imgPixelsColMaj = new float[imgSize];

        for (int i = 0; i < pics.length; i++) {
            try {

                toColMjr(grayScale(ImageIO.read(pics[i])).getData(), imgPixelsColMaj);

                pixelsGPU.getSubArray(i / depth).getLayer(i % depth).set(handle, imgPixelsColMaj);

            } catch (IOException e) {
                throw new IllegalArgumentException("Error reading image file: " + pics[i].getName(), e);
            }
        }

        return pixelsGPU;
    }

    /**
     * creates a rayscale version of an image if the image is not.
     *
     * @param image The image that may or may not be grayscale.
     * @return The original image if it is grayscale, and a grayscale copy if it
     * is not.
     */
    private static BufferedImage grayScale(BufferedImage image) {
        if (image == null) {
            throw new IllegalArgumentException("Could not open image file.");
        }

        if (image.getType() != BufferedImage.TYPE_BYTE_GRAY) {
            BufferedImage grayscaleImage = new BufferedImage(
                    image.getWidth(),
                    image.getHeight(),
                    BufferedImage.TYPE_BYTE_GRAY);
            grayscaleImage.getGraphics().drawImage(image, 0, 0, null);
            image = grayscaleImage;
        }

        return image;
    }

    /**
     * @brief Combines a folder of images into an ImagePlus object with a
     * specified depth (number of images per frame).
     *
     *
     * @param folderPath The path to the folder containing the image files. The
     * number of images in the folder should be a multiple of depth.
     * @param depth The number of images to include in each frame of the
     * ImagePlus stack.
     * @return An ImagePlus object representing the combined image stack, or
     * null if the folder is empty or invalid.
     * @throws IllegalArgumentException If the depth is less than or equal to
     * zero.
     */
    public static ImagePlus imagePlus(String folderPath, int depth) {
        if (depth <= 0) {
            throw new IllegalArgumentException("Depth must be greater than zero.");
        }

        File[] files = getImageFiles(folderPath);

        Opener opener = new Opener();

        ImagePlus img = opener.openImage(files[0].getAbsolutePath());

        ImageStack frames = new ImageStack(img.getWidth(), img.getHeight());

        for (int frameIndex = 0; frameIndex < files.length / depth; frameIndex++) {
            ImageStack layers = new ImageStack(img.getWidth(), img.getHeight());
            for (int layerIndex = 0; layerIndex < depth; layerIndex++) {
                img = opener.openImage(files[frameIndex * depth + layerIndex].getAbsolutePath());
                layers.addSlice(img.getProcessor());
            }
            frames.addSlice("frame " + frameIndex, layers.getProcessor(1)); // Add the completed frame

        }
        return new ImagePlus("Combined Image Stack", frames);
    }

    /**
     * Uses the suffix of the string to determine if it describes a picture
     * file. Recognized suffixes include .tif, .jpg, .jpeg, and .png.
     *
     * @param fileName The name of the file.
     * @return True if the suffix is for a picture file and false otherwise.
     */
    private static boolean isPicture(File fileName) {
        try {
            String suffix = fileName.getName()
                    .toLowerCase()
                    .substring(
                            fileName.getName().lastIndexOf(".") + 1
                    );

            return suffix.equals("tif")
                    || suffix.equals("jpg")
                    || suffix.equals("jpeg")
                    || suffix.equals("png");
        } catch (IndexOutOfBoundsException ex) {
            return false;
        }
    }

    /**
     * Gets all the image files in the proffered directory.
     *
     * @param parentDirectory The directory with the desired image files.
     * @return All the image files in the directory.
     */
    private static File[] getImageFiles(String parentDirectory) {
        File folder = new File(parentDirectory);

        if (!folder.exists() || !folder.isDirectory()) {
            throw new IllegalArgumentException("The provided path is not a valid folder.");
        }

        File[] imageFiles = folder.listFiles(name -> isPicture(name));

        if (imageFiles == null || imageFiles.length == 0) {
            throw new IllegalArgumentException("No image files found in the specified folder.");
        }

        java.util.Arrays.sort(imageFiles);

        return imageFiles;
    }

}
