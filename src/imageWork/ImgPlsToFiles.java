package imageWork;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.FileSaver;
import java.io.File;

/**
 *
 * @author E. Dov Neimand
 */
public class ImgPlsToFiles {//TODO: This class does not seem to work.

    /**
     * Saves each slice in each frame of an ImagePlus to a separate image
     * file.The saved files will be named based on the original filename, frame
     * number, and slice number. The images will be saved in the same directory
     * as the original image.
     *
     * @param imp The ImagePlus object to process.
     * @param saveTo The folder the file should be saved in.
     */
    public static void saveSlices(ImagePlus imp, String saveTo) {

        String fileName = imp.getTitle();

        ImageStack stack = imp.getStack();

        int numFrames = imp.getNFrames();
        int numSlices = imp.getNSlices();

        for (int frame = 1; frame <= numFrames; frame++) {
            for (int slice = 1; slice <= numSlices; slice++) {
                int stackIndex = imp.getStackIndex(1, slice, frame); // Channel is always 1 for individual slices
                ImagePlus sliceImp = new ImagePlus(fileName + "_F" + frame + "_Z" + slice, stack.getProcessor(stackIndex));

                FileSaver fs = new FileSaver(sliceImp);
                
                String fullPath = new File(saveTo, fileName + sliceImp.getShortTitle()).getAbsolutePath();
                
                fs.saveAsTiff(fullPath);

                System.out.println("imageWork.ImgPlsToFiles.saveSlices - Saved: " + fullPath);
            }
        }

    }
}
