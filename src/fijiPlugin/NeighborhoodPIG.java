package fijiPlugin;

import FijiInput.UserInput;

import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import imageWork.ColorHeatMapCreator;
import imageWork.GrayScaleHeatMapCreator;
import imageWork.HeatMapCreator;
import imageWork.VectorImg;
import java.util.Arrays;
import main.Test;

/**
 * Each neighborhood pig has it's own handle.
 *
 * @author E. Dov Neimand
 */
public class NeighborhoodPIG implements AutoCloseable {

    public final StructureTensorMatrices stm; //TODO: this should probably be made private

    public final static boolean D3 = true, D2 = false;

    private final String[] sourceFileNames;
    
    private Handle handle;
    
    /**
     *
     * @param handle The context.
     * @param image An image.
     * @param ui The input from the user.
     */
    public NeighborhoodPIG(Handle handle, ImagePlus image, UserInput ui) {
        this.handle = handle;

        this.sourceFileNames = image.getImageStack().getSliceLabels();
        
        stm = new StructureTensorMatrices(handle, image, ui);

    }


    /**
     * A heat map of the orientation in the xy plane.
     *
     * @param color True for a color image, false for grayscale.
     * @param tolerance When is a value considered 0.
     * @return A heat map of the orientation in the xy plane.
     */
    public HeatMapCreator getAzimuthalAngles(boolean color, double tolerance) {

        return color ? new ColorHeatMapCreator(handle,
                concat(sourceFileNames, " Azimuth"),
                "Azimuth Angle Heatmap",
                stm.azimuthAngle(),
                stm.getCoherence(),
                stm.downSampled
        ) : new GrayScaleHeatMapCreator(
                concat(sourceFileNames, " Azimuth"),
                "Azimuth Angle Heatmap",
                handle,
                stm.azimuthAngle(),
                stm.getCoherence(),
                tolerance,
                stm.downSampled
        );
    }

    /**
     * A heat map of the orientation in the yz plane.
     *
     * @param color True for a color image, false for grayscale.
     * @param tolerance
     * @return A heat map of the zenith angles.
     */
    public HeatMapCreator getZenithAngles(boolean color, double tolerance) {
        return color ? new ColorHeatMapCreator(
                handle,
                concat(sourceFileNames, " Zenith Angle"),
                "Zenith Angle Heatmap",
                stm.zenithAngle(),
                stm.getCoherence(),
                stm.downSampled
        )
                : new GrayScaleHeatMapCreator(
                        concat(sourceFileNames, " Zenith Angle"),
                        "Zenith Angle Heatmap",
                        handle,
                        stm.zenithAngle(),
                        stm.getCoherence(),
                        tolerance,
                        stm.downSampled
                );
    }

    /**
     * The coherence heatmap.
     *
     * @return The coherence heatmap.
     */
    public HeatMapCreator getCoherence() {
        return new GrayScaleHeatMapCreator(
                concat(sourceFileNames, " coherence"),
                "Coherence",
                handle,
                stm.getCoherence(),
                null,
                0,
                stm.dim
        );
    }

    /**
     * An image of all the nematic vectors
     *
     * @param spacing The space between the vectors.
     * @param vecMag The magnitude of the vectors to be drawn.
     * @param useCoherence True to use coherence, false to set all vector
     * intensities to 1.
     * @return An image of all the nematic vectors
     */
    public ImagePlus getVectorImg(int spacing, int vecMag, boolean useCoherence) {

        return new VectorImg(
                handle,
                stm.dim,
                vecMag,
                stm.getVectors(),
                stm.getCoherence(),
                spacing, useCoherence, 0.01
        ).get();
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

    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        stm.close();
    }
}
    


//
//
//    /**
//     * A factory method for a neighborhood pig.
//     *
//     * @param handle
//     * @param imp The imagePlus from which the image data is taken.
//     * @param ui Input submited by the user.
//     * @return A neighborhoodPIG.
//     */
//    public static NeighborhoodPIG get(Handle handle, ImagePlus imp, UserInput ui) {
//    
//
//            return new NeighborhoodPIG(
//                    handle,
//                    gpuImmage,
//                    imp.getImageStack().getSliceLabels(),
//                    ui
//            );
//        
//
//    }




//    /**
//     * A factory method for a neighborhood pig.
//     *
//     * @param handle
//     * @param folderPath All images in the folder should have the same height
//     * and width.
//     * @param ui User set specifications.
//     * @param depth
//     * @return A neighborhoodPIG.
//     */
//    public static NeighborhoodPIG getWithIJ(Handle handle, String folderPath, int depth, UserInput ui) {
//
//        ImagePlus ip = ProcessImage.imagePlus(folderPath, depth);
//
//        try (PArray2dToD2d gpuImage = ProcessImage.processImages(handle, ip, ui)) {
//
//            return new NeighborhoodPIG(
//                    handle,
//                    gpuImage,
//                    new File(folderPath).list(),
//                    ui
//            );
//        }
//
//    }
//    
//
//
//
//    /**
//     * A factory method for a neighborhood pig.
//     *
//     * @param handle
//     * @param folderPath All images in the folder should have the sameheight and
//     * width.
//     * @param ui The input from the user.
//     * @param depth
//     * @return A neighborhoodPIG.
//     */
//    public static NeighborhoodPIG get(Handle handle, String folderPath, int depth, UserInput ui) {
//        try {
//
//            File[] imageFiles = ProcessImage.getImageFiles(folderPath);
//
//            BufferedImage firstImage = ImageIO.read(imageFiles[0]);
//
//            int width = ui.downSample(firstImage.getWidth()),
//                    height = ui.downSample(firstImage.getHeight());
//
//            try (PArray2dToD2d gpuImage = ProcessImage.processImages(handle, imageFiles, height, width, depth)) {
//
//                return new NeighborhoodPIG(
//                        handle,
//                        gpuImage,
//                        Arrays.stream(imageFiles).map(File::getName).toArray(String[]::new),
//                        ui
//                );
//            }
//        } catch (IOException ex) {
//            Logger.getLogger(NeighborhoodPIG.class.getName()).log(Level.SEVERE, null, ex);
//            throw new RuntimeException();
//        }
//    }
//
//
//}
