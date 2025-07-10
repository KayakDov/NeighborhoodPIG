package fijiPlugin;

import FijiInput.UserInput;

import JCudaWrapper.resourceManagement.Handle;
import imageWork.HeatMapCreator;
import imageWork.MyImagePlus;
import imageWork.vectors.ColorVectorImg;
import imageWork.vectors.VectorImg;
import imageWork.vectors.WhiteVectorImg;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;

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
    public NeighborhoodPIG(Handle handle, MyImagePlus image, UserInput ui) {
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
    public HeatMapCreator getAzimuthalAngles(double tolerance) {

        return new HeatMapCreator(
                concat(sourceFileNames, " Azimuth"),
                "Azimuth Angle Heatmap",
                handle,
                stm.azimuthAngle(),
                stm.getCoherence(),
                tolerance
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
        return new HeatMapCreator(
                concat(sourceFileNames, " Zenith Angle"),
                "Zenith Angle Heatmap",
                handle,
                stm.zenithAngle(),
                stm.getCoherence(),
                tolerance
        );
    }

    /**
     * The coherence heatmap.
     *
     * @return The coherence heatmap.
     */
    public HeatMapCreator getCoherence(double tolerance) {
        return new HeatMapCreator(
                concat(sourceFileNames, " coherence"),
                "Coherence",
                handle,
                stm.getCoherence(),
                null,
                tolerance
        );
    }

    /**
     * An image of all the nematic vectors
     * TODO; Create user coherence parameter and pass it here.
     * @param spacingXY The space between the vectors in the xy dimension.
     * @param spacingZ The space between the vectors in the z dimension.
     * @param vecMag The magnitude of the vectors to be drawn.
     * @param useCoherence True to use coherence, false to set all vector
     * intensities to 1.
     * @param overlay The dimensions of the base image to be overlaid. Leave
     * this null if overlay is false;
     * @param color true if the vectors should be colored, false if they should be white.
     * @return An image of all the nematic vectors
     */
    public VectorImg getVectorImg(int spacingXY, int spacingZ, int vecMag, boolean useCoherence, Dimensions overlay, boolean color) {

        return color?
                new ColorVectorImg(overlay, handle, vecMag, stm.getVectors(), stm.getCoherence(), spacingXY, spacingZ, .01):
                new WhiteVectorImg(overlay, handle, vecMag, stm.getVectors(), stm.getCoherence(), spacingXY, spacingZ, .01);
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
