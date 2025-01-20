package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3StrideDim;
import JCudaWrapper.array.DArray;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.Opener;
import ij.process.ImageProcessor;
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
public class NeighborhoodPIG extends TensorOrd3StrideDim implements AutoCloseable {

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
    private NeighborhoodPIG(TensorOrd3Stride image, String[] sourceFileNames, int neighborhoodSize, double tolerance) {
        super(image);

        this.sourceFileNames = sourceFileNames == null ? defaultNames() : sourceFileNames;

        try (Gradient grad = new Gradient(image, handle)) {            
            stm = new StructureTensorMatrix(grad, neighborhoodSize, tolerance);
        }
    }

    public String[] defaultNames() {
        String[] names = new String[depth * batchSize];
        int nameIndex = 0;
        for (int frameInd = 0; frameInd < batchSize; frameInd++)
            for (int layerInd = 0; layerInd < depth; layerInd++)
                names[nameIndex++] = "Frame " + frameInd + " Layer " + layerInd;
        return names;

    }

    /**
     * A heat map of the orientation in the xy plane.
     * @param useCoherence True if pixel intensity should be tied to orientation confidence (coherence)
     * @return A heat map of the orientation in the xy plane.
     */
    public ImageCreator getImageOrientationXY(boolean useCoherence) {

        return new ImageCreator(handle, concat(sourceFileNames, " XY orientation"), stm.getOrientationXY(), useCoherence? stm.getCoherence(): null);
    }

    /**
     * A heat map of the orientation in the yz plane.
     *
     * @param useCoherence True if pixel intensity should be tied to orientation confidence (coherence)
     * @return A heat map of the orientation in the yz plane.
     */
    public ImageCreator getImageOrientationYZ(boolean useCoherence) {
        return new ImageCreator(handle, concat(sourceFileNames, " YZ orientation"), stm.getOrientationXY(), useCoherence? stm.getCoherence(): null);
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
            if (nc == null) return concatination;
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
     * @param neighborhoodR The radius of a neighborhood. The neighborhood will
     * be a square and the radius is the distance from the center to the nearest
     * edge.
     * @param tolerance Close enough to 0.
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG get(Handle handle, ImagePlus imp, int neighborhoodR, double tolerance) {//TODO: get image names

        try (DArray gpuImmage = processImages(handle, imp)) {

            return new NeighborhoodPIG(
                    new TensorOrd3Stride(handle,
                            imp.getHeight(),
                            imp.getWidth(),
                            imp.getNSlices() / imp.getNFrames(),
                            imp.getNFrames(), 
                            gpuImmage
                    ),
                    imp.getImageStack().getSliceLabels(),
                    neighborhoodR,
                    tolerance
            );
        }

    }

    /**
     * A factory method for a neighborhood pig.
     *
     * @param handle
     * @param folderPath All images in the folder should have the sameheight and
     * width.
     * @param neighborhoodR The radius of a neighborhood. The neighborhood will
     * be a square and the radius is the distance from the center to the nearest
     * edge.
     * @param depth
     * @param tolerance Close enough to 0.
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG get(Handle handle, String folderPath, int depth, int neighborhoodR, double tolerance) {
        try {

            File[] imageFiles = getImageFiles(folderPath);

            BufferedImage firstImage = ImageIO.read(imageFiles[0]);

            try (DArray gpuImage = processImages(handle, imageFiles, firstImage.getHeight(), firstImage.getWidth())) {

                return new NeighborhoodPIG(
                        new TensorOrd3Stride(handle,
                                firstImage.getHeight(),
                                firstImage.getWidth(),
                                depth,
                                imageFiles.length / depth,
                                gpuImage
                        ),
                        new File(folderPath).list(),
                        neighborhoodR,
                        tolerance
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
     * @param neighborhoodR The radius of a neighborhood. The neighborhood will
     * be a square and the radius is the distance from the center to the nearest
     * edge.
     * @param depth
     * @param tolerance Close enough to 0.
     * @return A neighborhoodPIG.
     */
    public static NeighborhoodPIG getWithIJ(Handle handle, String folderPath, int depth, int neighborhoodR, double tolerance) {

        ImagePlus ip = imagePlus(folderPath, depth);
        try (DArray gpuImage = processImages(handle, ip)) {

            return new NeighborhoodPIG(
                    new TensorOrd3Stride(handle,
                            ip.getHeight(),
                            ip.getWidth(),
                            depth,
                            ip.getNFrames(),
                            gpuImage
                    ),
                    new File(folderPath).list(),
                    neighborhoodR,
                    tolerance
            );
        }

    }

    /**
     * Converts a grayscale or RGB ImagePlus object into a single column-major
     * GPU array of pixel values. RGB images are converted to grayscale first.
     *
     * @param imp The input ImagePlus object.
     * @param handle Context.
     * @return A single column-major GPU array containing pixel values of all
     * slices.
     */
    public final static DArray processImages(Handle handle, ImagePlus imp) {

        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32){
            System.out.println("fijiPlugin.NeighborhoodPIG.processImages(): Converting image to grayscale.");
            IJ.run(imp, "32-bit", "");
        }

        ImageStack stack = imp.getStack();
        
        DArray processedImage = DArray.empty(stack.getWidth() * stack.getHeight() * stack.getSize());
        
        int imgSize = stack.getHeight()*stack.getWidth();
        
        double[] columnMajorSlice = new double[imgSize];

        for (int slice = 1; slice <= stack.getSize(); slice++) {
            ImageProcessor ip = stack.getProcessor(slice);
            float[][] pixels = ip.getFloatArray();

            for (int col = 0; col < imp.getWidth(); col++)
                for (int row = 0; row < imp.getHeight(); row++)
                    columnMajorSlice[col*stack.getHeight() + row] = pixels[col][row];
            
            processedImage.set(handle, columnMajorSlice, (slice - 1)*imgSize);
        }

        return processedImage;
    }

    /**
     * Copies the raster to an array in column major order.
     *
     * @param raster The raster being written from.
     * @param writeTo The array being written to.
     */
    private static void toColMjr(Raster raster, double[] writeTo) {
        for (int col = 0; col < raster.getWidth(); col++)
            for (int row = 0; row < raster.getHeight(); row++)
                writeTo[col * raster.getHeight() + row] = raster.getSample(col, row, 0);
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
     *
     * @return A single column-major GPU array containing pixel values of all
     * images.
     * @throws IllegalArgumentException If no valid images are found in the
     * folder.
     */
    public final static DArray processImages(Handle handle, File[] pics, int height, int width) {

        int numPixels = height * width * pics.length;
        DArray pixelsGPU = DArray.empty(numPixels);

        int imgSize = width * height;
        double[] imgPixelsColMaj = new double[imgSize];

        int pixelInd = 0;

        for (File file : pics) {
            try {

                toColMjr(grayScale(ImageIO.read(file)).getData(), imgPixelsColMaj);

                pixelsGPU.set(handle, imgPixelsColMaj, (pixelInd / imgSize) * imgSize);

                pixelInd += imgSize;

            } catch (IOException e) {
                throw new IllegalArgumentException("Error reading image file: " + file.getName(), e);
            }
        }
        // Create and return the GPU array
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
        if (image == null)
            throw new IllegalArgumentException("Could not open image file.");

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
        if (depth <= 0)
            throw new IllegalArgumentException("Depth must be greater than zero.");

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
        String suffix = fileName.getName()
                .toLowerCase()
                .substring(
                        fileName.getName().lastIndexOf(".")
                );

        return suffix.equals(".tif")
                || suffix.equals(".jpg")
                || suffix.equals(".jpeg")
                || suffix.equals(".png");
    }

    /**
     * Gets all the image files in the proffered directory.
     *
     * @param parentDirectory The directory with the desired image files.
     * @return All the image files in the directory.
     */
    private static File[] getImageFiles(String parentDirectory) {
        File folder = new File(parentDirectory);

        if (!folder.exists() || !folder.isDirectory())
            throw new IllegalArgumentException("The provided path is not a valid folder.");

        File[] imageFiles = folder.listFiles(name -> isPicture(name));

        if (imageFiles == null || imageFiles.length == 0)
            throw new IllegalArgumentException("No image files found in the specified folder.");

        java.util.Arrays.sort(imageFiles);

        return imageFiles;
    }

}
