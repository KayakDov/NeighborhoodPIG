package imageWork;

import FijiInput.UsrInput;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.Opener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;

/**
 *
 * @author E. Dov Neimand
 */
public class ProcessImage {

    /**
     * @brief Combines a folder of images into an ImagePlus object with a
     * specified depth (number of images per frame).
     *
     *
     * @param folderPath The path to the folder containing the image files. The
     * number of images in the folder should be a multiple of depth.
     * @param depth The number of images to include in each frame of the
     * ImagePlus stack. This must be greater than 0.
     * @return An ImagePlus object representing the combined image stack, or
     * null if the folder is empty or invalid.
     * @throws IllegalArgumentException If the depth is less than or equal to
     * zero.
     */
    public static ImagePlus getImagePlus(String folderPath, int depth) {

        File[] files = getImageFiles(folderPath);

        Opener opener = new Opener();
        
        ImagePlus imp = opener.openImage(files[0].getPath());
                
        Dimensions dim = new Dimensions(imp.getHeight(), imp.getWidth(), depth, files.length/depth);

        ImageStack frameSequence = dim.emptyStack();

        for (File file : files) {
            ImagePlus currentImp = opener.openImage(file.getAbsolutePath());
            frameSequence.addSlice(currentImp.getProcessor());
        }

        return dim.setToHyperStack(new ImagePlus(folderPath.substring(Math.max(folderPath.lastIndexOf(File.pathSeparator), 0)), frameSequence));
    }

    /**
     * Uses the suffix of the string to determine if it describes a picture
     * file. Recognized suffixes include .tif, .jpg, .jpeg, and .png.
     *
     * @param fileName The name of the file.
     * @return True if the suffix is for a picture file and false otherwise.
     */
    public static boolean isPicture(File fileName) {
        try {
            String suffix = fileName.getName().toLowerCase().substring(fileName.getName().lastIndexOf(".") + 1);
            return suffix.equals("tif") || suffix.equals("jpg") || suffix.equals("jpeg") || suffix.equals("png");
        } catch (IndexOutOfBoundsException ex) {
            return false;
        }
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
            BufferedImage grayscaleImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
            grayscaleImage.getGraphics().drawImage(image, 0, 0, null);
            image = grayscaleImage;
        }
        return image;
    }

    /**
     * Processes a hyperstack and returns a PArray2dToD2d containing the
     * processed image data. The 2d array of pointers points to the layers of
     * the frames, where each column is a frame and each row is a layer of that
     * frame.
     *
     * @param handle The handle used for FArray operations.
     * @param imp The ImagePlus object representing the hyperstack.
     * @param ui Down sampling requests are used to potentially shave off a bit
     * of the image so that the image dimensions are divisible by the down
     * sample factor.
     * @return A FArray containing the image data in column-major order for all
     * frames, slices, and channels.
     */
    public static final P2dToF2d processImages(Handle handle, MyImagePlus imp, UsrInput ui) {

        // Convert the image to grayscale if necessary
        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) {
            System.out.println("fijiPlugin.NeighborhoodPIG.processImages(): Converting image to grayscale.");
            IJ.run(imp, "32-bit", "");
        }
        Dimensions dim = new Dimensions(imp);        

        P2dToF2d processedImage = new P2dToF2d(dim.depth, dim.batchSize, dim.height, dim.width, handle);
        float[] columnMajorSlice = new float[dim.layerSize()];

        for (int frame = 1; frame <= dim.batchSize; frame++) {
            for (int slice = 1; slice <= dim.depth; slice++) {

                imp.setPosition(1, slice, frame);

                float[][] pixels = imp.getProcessor().getFloatArray();

                for (int col = 0; col < dim.width; col++)
                    System.arraycopy(pixels[col], 0, columnMajorSlice, col * dim.height, dim.height);

                processedImage.get(slice - 1, frame - 1).getVal(handle).set(handle, columnMajorSlice);

            }
        }

        processedImage.scale(handle, 1f / 255);

        return processedImage;
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
            throw new IllegalArgumentException("The provided path " + parentDirectory + " is not a valid folder.");
        }
        File[] imageFiles = folder.listFiles(name -> ProcessImage.isPicture(name));
        if (imageFiles == null || imageFiles.length == 0) {
            throw new IllegalArgumentException("No image files found in the specified folder.");
        }
        Arrays.sort(imageFiles);
        return imageFiles;
    }

}
