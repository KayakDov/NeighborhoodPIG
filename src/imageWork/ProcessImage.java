package imageWork;

import FijiInput.UserInput;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
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
import javax.imageio.ImageIO;

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
     * Processes a hyperstack and returns a FArray containing the processed
     * image data.
     *
     * @param handle The handle used for FArray operations.
     * @param imp The ImagePlus object representing the hyperstack.
     * @param ui Down sampling requests are used to potentially shave off a bit
     * of the image so that the image dimensions are divisible by the down
     * sample factor.
     * @return A FArray containing the image data in column-major order for all
     * frames, slices, and channels.
     */
    public static final PArray2dToD2d processImages(Handle handle, ImagePlus imp, UserInput ui) {
        // Convert the image to grayscale if necessary
        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) {
            System.out.println("fijiPlugin.NeighborhoodPIG.processImages(): Converting image to grayscale.");
            IJ.run(imp, "32-bit", "");
        }
        int width = ui.downSample(imp.getWidth());
        int height = ui.downSample(imp.getHeight());
        int channels = imp.getNChannels();
        int depth = imp.getNSlices();
        int frames = imp.getNFrames();
        int imgSize = width * height;
        
        PArray2dToD2d processedImage = new PArray2dToD2d(depth, frames, height, width);
        
        double[] columnMajorSlice = new double[imgSize];
        // Iterate over frames, slices, and channels
        for (int frame = 1; frame <= frames; frame++) {
            for (int slice = 1; slice <= depth; slice++) {
                for (int channel = 1; channel <= channels; channel++) {
                    imp.setPosition(channel, slice, frame);
                    ImageProcessor ip = imp.getProcessor();
                    float[][] pixels = ip.getFloatArray();
                    for (int col = 0; col < width; col++)
                        System.arraycopy(pixels[col], 0, columnMajorSlice, col * height, height);

                    processedImage.get(slice - 1, frame - 1).set(handle, new DArray2d(height, width).set(handle, columnMajorSlice));
                }
            }
        }
        return processedImage;
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
    public static final PArray2dToD2d processManyImages(Handle handle, ImagePlus imp, UserInput ui) {
        // Convert the image to grayscale if necessary
        if (imp.getType() != ImagePlus.GRAY8 && imp.getType() != ImagePlus.GRAY16 && imp.getType() != ImagePlus.GRAY32) {
            System.out.println("fijiPlugin.NeighborhoodPIG.processImages(): Converting image to grayscale.");
            IJ.run(imp, "32-bit", "");
        }
        int width = ui.downSample(imp.getWidth());
        int height = ui.downSample(imp.getHeight());
        int channels = imp.getNChannels();
        int depth = imp.getNSlices();
        int frames = imp.getNFrames();
        int imgSize = width * height;

        PArray2dToD2d processedImage = new PArray2dToD2d(depth, frames, height, width);
        double[] columnMajorSlice = new double[imgSize];

        for (int frame = 1; frame <= frames; frame++) {
            for (int slice = 1; slice <= depth; slice++) {
                for (int channel = 1; channel <= channels; channel++) {
                    imp.setPosition(channel, slice, frame);

                    float[][] pixels = imp.getProcessor().getFloatArray();

                    for(int col = 0; col < width; col++)
                        for(int row = 0; row < height; row++)
                            columnMajorSlice[col*height + row] = pixels[col][row];

                    DArray2d gpuSlice = new DArray2d(height, width).set(handle, columnMajorSlice);

                    System.out.println("imageWork.ProcessImage.processManyImages()");
                    
                    processedImage.get(slice - 1, frame - 1).setVal(handle, gpuSlice);
                    
                    
                }
            }
        }
        return processedImage;
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
    public static final PArray2dToD2d processImages(Handle handle, File[] pics, int height, int width, int depth) {

        PArray2dToD2d pixelsGPU = new PArray2dToD2d(depth, pics.length / depth, height, width);
        
        double[] imgPixelsColMaj = new double[width * height];
        
        for (int i = 0; i < pics.length; i++) {
            try {
                BufferedImage bi = ImageIO.read(pics[i]);
                if (width < bi.getWidth() || height < bi.getHeight()) bi = bi.getSubimage(0, 0, width, height);
                
                toColMjr(grayScale(bi).getData(), imgPixelsColMaj);
                
                pixelsGPU.get(i % depth, i / depth).set(handle, new DArray2d(height, width).set(handle, imgPixelsColMaj));
            
            } catch (IOException e) {
                throw new IllegalArgumentException("Error reading image file: " + pics[i].getName(), e);
            }
        }
        return pixelsGPU;
    }

    /**
     * Gets all the image files in the proffered directory.
     *
     * @param parentDirectory The directory with the desired image files.
     * @return All the image files in the directory.
     */
    public static File[] getImageFiles(String parentDirectory) {
        File folder = new File(parentDirectory);
        if (!folder.exists() || !folder.isDirectory()) {
            throw new IllegalArgumentException("The provided path is not a valid folder.");
        }
        File[] imageFiles = folder.listFiles(name -> ProcessImage.isPicture(name));
        if (imageFiles == null || imageFiles.length == 0) {
            throw new IllegalArgumentException("No image files found in the specified folder.");
        }
        Arrays.sort(imageFiles);
        return imageFiles;
    }

}
