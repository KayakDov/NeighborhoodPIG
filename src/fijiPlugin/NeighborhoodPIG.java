package fijiPlugin;

import JCudaWrapper.algebra.TensorOrd3Stride;
import JCudaWrapper.algebra.TensorOrd3dStrideDim;
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
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG extends TensorOrd3dStrideDim implements AutoCloseable {

    private StructureTensorMatrix stm;

    public final static boolean D3 = true, D2 = false;

    /**
     *
     * @param image An image.
     * @param neighborhoodSize The size of the edges of each neighborhood
     * square.
     * @param tolerance How close must a number be to 0 to be considered 0.
     */
    public NeighborhoodPIG(TensorOrd3Stride image, int neighborhoodSize, double tolerance) {
        super(image);

        try (Gradient grad = new Gradient(image, handle)) {
            //        image.close();
            stm = new StructureTensorMatrix(grad, neighborhoodSize, tolerance);
        }
    }

    /**
     * A heat map of the orientation in the xy plane.
     *
     * @return A heat map of the orientation in the xy plane.
     */
    public ImageCreator getImageOrientationXY() {
        return new ImageCreator(handle, stm.getOrientationXY(), stm.getCoherence());
    }

    /**
     * A heat map of the orientation in the yz plane.
     *
     * @return A heat map of the orientation in the yz plane.
     */
    public ImageCreator getImageOrientationYZ() {
        return new ImageCreator(handle, stm.getOrientationXY(), stm.getCoherence());
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
    public static NeighborhoodPIG get(Handle handle, ImagePlus imp, int neighborhoodR, double tolerance) {
        try (DArray gpuImage = processImages(handle, imp)) {
            return new NeighborhoodPIG(
                    new TensorOrd3Stride(handle,
                            imp.getHeight(),
                            imp.getWidth(),
                            imp.getNSlices() / imp.getNFrames(),
                            imp.getNFrames()
                    ),
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

            try (DArray gpuImage = processImages(handle, imageFiles)) {

                return new NeighborhoodPIG(
                        new TensorOrd3Stride(handle,
                                firstImage.getHeight(),
                                firstImage.getWidth(),
                                depth,
                                imageFiles.length / depth,
                                gpuImage
                        ),
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

        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32)
            IJ.run(imp, "32-bit", "");

        ImageStack stack = imp.getStack();
        double[] columnMajorArray = new double[stack.getWidth() * stack.getHeight() * stack.getSize()];

        int index = 0;

        for (int slice = 1; slice <= stack.getSize(); slice++) {
            ImageProcessor ip = stack.getProcessor(slice);
            float[][] pixels = ip.getFloatArray();

            for (int col = 0; col < imp.getWidth(); col++)
                for (int row = 0; row < imp.getHeight(); row++)
                    columnMajorArray[index++] = pixels[col][row];
        }

        return new DArray(handle, columnMajorArray);
    }

    /**
     * Converts grayscale or RGB image files in a folder into a single
     * column-major GPU array of pixel values. RGB images are converted to
     * grayscale first.
     *
     * @param handle Context.
     * @param pics Path to the folder containing image files.
     * @return A single column-major GPU array containing pixel values of all
     * images.
     * @throws IllegalArgumentException If no valid images are found in the
     * folder.
     */
    public final static DArray processImages(Handle handle, File[] pics) {

        List<Double> pixelValues = new ArrayList<>();

        for (File file : pics) {
            try {
                BufferedImage image = grayScale(ImageIO.read(file));

                if (image == null)
                    throw new IllegalArgumentException("Could not open image file: " + file.getName());

                Raster raster = image.getData();
                int width = image.getWidth();
                int height = image.getHeight();

                for (int col = 0; col < width; col++)
                    for (int row = 0; row < height; row++)
                        pixelValues.add((double) raster.getSample(col, row, 0));

            } catch (IOException e) {
                throw new IllegalArgumentException("Error reading image file: " + file.getName(), e);
            }
        }

        double[] columnMajorArray = pixelValues.stream().mapToDouble(Double::doubleValue).toArray();

        // Create and return the GPU array
        return new DArray(handle, columnMajorArray);
    }

    /**
     * creates a rayscale version of an image if the image is not.
     *
     * @param image The image that may or may not be grayscale.
     * @return The original image if it is grayscale, and a grayscale copy if it
     * is not.
     */
    private static BufferedImage grayScale(BufferedImage image) {
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

        ImageStack frameStack = new ImageStack(img.getWidth(), img.getHeight());
        ImageStack frame = new ImageStack(img.getWidth(), img.getHeight());

        for (int frameIndex = 0; frameIndex < files.length/depth; frameIndex++) {
            for (int layer = 0; layer < depth; layer++) {
                img = opener.openImage(files[frameIndex].getAbsolutePath());
                frame.addSlice(img.getProcessor());
            }
            frameStack.addSlice("frame " + frameIndex, frame.getProcessor(1)); // Add the completed frame

        }

        return new ImagePlus("Combined Image Stack", frameStack);
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

        File[] imageFiles = folder.listFiles((dir, name)
                -> name.toLowerCase().endsWith(".tif")
                || name.toLowerCase().endsWith(".jpg")
                || name.toLowerCase().endsWith(".jpeg")
                || name.toLowerCase().endsWith(".png"));

        if (imageFiles == null || imageFiles.length == 0)
            throw new IllegalArgumentException("No image files found in the specified folder.");

        java.util.Arrays.sort(imageFiles);

        return imageFiles;
    }
}
