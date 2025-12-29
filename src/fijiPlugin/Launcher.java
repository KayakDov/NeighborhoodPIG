package fijiPlugin;

import FijiInput.UsrInput;
import FijiInput.field.VF;
import JCudaWrapper.array.Array;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.resourceManagement.GPU;
import JCudaWrapper.resourceManagement.Handle;
import ij.IJ;
import imageWork.HeatMapCreator;
import imageWork.MyImagePlus;
import imageWork.MyImageStack;
import imageWork.TxtSaver;
import imageWork.vectors.VectorImg;
import java.awt.Color;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Level;
import java.util.logging.Logger;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class Launcher implements Runnable {

    /**
     * Defines the available save formats for plugin outputs.
     * <ul>
     * <li>{@code tiff}: Save output images as TIFF files.</li>
     * <li>{@code png}: Save output images as PNG files.</li>
     * <li>{@code fiji}: Display outputs directly within Fiji/ImageJ.</li>
     * <li>{@code txt}: Save raw vector data (x, y, z, nx, ny, nz) to a text
     * file.</li>
     * </ul>
     */
    public enum Save {
        tiff, png, fiji, txt // Added txt for saving raw vector data
    }

    public final MyImageStack vf, coh, az, zen;
    public final ExecutorService es = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());
    private final UsrInput ui;
    private final Save save;

    /**
     * Constructs the launcher.
     *
     * @param ui The user input.
     * @param save How the data should be saved.
     */
    public Launcher(UsrInput ui, Save save) {
        vf = VectorImg.space(ui.dim, ui.spacingXY.orElse(1), ui.spacingZ.orElse(1), ui.vfMag.orElse(1), ui.overlay.orElse(false) ? ui.img.dim() : null).emptyStack();
        coh = ui.dim.emptyStack();
        az = ui.dim.emptyStack();
        zen = ui.dim.emptyStack();

        this.ui = ui;
        this.save = save;
    }

    /**
     * The maximum number of frames that can be processed at once. TODO: this
     * should take downsampling into account!
     *
     * @return The maximum number of frames that can be processed at once.
     */
    public int framesPerRun() {

        long freeMemory = GPU.freeMemory();

        if (freeMemory == 0)
            throw new RuntimeException("There is no free GPU memory.");

        long voxlesPerFrame = (long) ui.dim.height * ui.dim.width * ui.dim.depth;

        int framesPerRun = (int) ((freeMemory / voxlesPerFrame) / (Sizeof.DOUBLE * (ui.dim.depth > 1 ? 6 : 3) + Sizeof.FLOAT * (ui.dim.depth > 1 ? 3 : 2)));

        if (framesPerRun > 1)
            return framesPerRun / 2;
        else if (framesPerRun <= 0)
            IJ.error("Your stack has a high depth relative to GPU size. This may cause a crash due to insufficiant GPU memory to process a complete frame.");

        return 1;

    }

    /**
     * Takes the results from running NeighborhoodPIG and loads them into
     * imageStacks or saved files based on the user's input. This method
     * orchestrates the creation of various output images such as heatmaps for
     * azimuthal and zenith angles, vector fields, and coherence maps. It also
     * handles the saving of raw vector data if specified by the user.
     *
     * @param ui The {@link UsrInput} object containing all the user-defined
     * parameters and preferences for output generation (e.g., whether to
     * generate heatmaps, vector fields, coherence, or save raw data).
     * @param fp The {@link FijiPlugin} instance, which provides access to
     * shared resources like dimensions ({@link FijiPlugin#dim}), image stacks
     * for different outputs (e.g., {@link FijiPlugin#az},
     * {@link FijiPlugin#zen}, {@link FijiPlugin#vf},
     * {@link FijiPlugin#coh}), and the executor service ({@link FijiPlugin#es})
     * for parallel processing.
     * @param handle A {@link Handle} object, typically representing a GPU
     * device handle or context, used for managing GPU memory and kernel
     * execution within the current processing iteration.
     * @param np The {@link NeighborhoodPIG} object, which contains the computed
     * results for the current image frame(s), including coherence, vector
     * types, and methods to retrieve azimuthal angles, zenith angles, and
     * vector images.
     * @return The depth of the generated vector image if a vector field was
     * created ({@code ui.vectorField.is()} is true); otherwise, returns 0. This
     * value is used subsequently for displaying or saving vector field results
     * correctly.
     */
    private int processNPIGResults(UsrInput ui, Handle handle, NeighborhoodPIG np) {

        int depth = 0;

        if (ui.heatMap) {

            appendHM(az, np.getAzimuthalAnglesHeatMap(ui.tolerance), 0, (float) Math.PI, es);
            if (ui.img.dim().hasDepth())
                appendHM(zen, np.getZenithAnglesHeatMap(false, 0.01), 0, (float) Math.PI, es);
        }

        if (ui.vectorField.is())
            depth = appendVF(
                    ui,
                    np.getVectorImg(
                            ui.spacingXY.get(),
                            ui.spacingZ.orElse(1),
                            ui.vfMag.get(),
                            false,
                            ui.overlay.orElse(false) ? ui.img.dim() : null,
                            ui.vectorField == VF.Color
                    ),
                    vf,
                    es
            );

        if (ui.coherence)
            appendHM(coh, np.getCoherenceHeatMap(ui.tolerance), 0, 1, es);

        if (ui.saveDatToDir.isPresent())
            new TxtSaver(ui.dim, np.stm.getVectors(), handle, ui.saveDatToDir.get(), ui.spacingXY.orElse(1), ui.spacingZ.orElse(1), np.stm.coherence, ui.tolerance).saveAllVectors();

        return depth;
    }

    /**
     * prints the amount of time since the clock started.
     *
     * @param startTime
     */
    private static void printTime(long startTime) {
        long endTime = System.currentTimeMillis();
        System.out.println("Execution time: " + (endTime - startTime) + " milliseconds");
    }

    /**
     * Waits for all open threads to close.
     *
     * @param threads All open threads.
     */
    private void awaitThreadTermination() {
        es.shutdown();

        try {
            es.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException ex) {
            Logger.getLogger(FijiPlugin.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Appends to the stack from the vector field.
     *
     * @param ui The user input.
     * @param vecImg The vector image to be added to the stack.
     * @param vf The stack to be appended to.
     * @param es The executor service.
     * @return The depth of the vector image.
     */
    public static int appendVF(UsrInput ui, VectorImg vecImg, MyImageStack vf, ExecutorService es) {
        vf.concat(vecImg.imgStack(es));
        return vecImg.getOutputDimensions().depth;
    }

    /**
     * Appends the heatmap to the stack.
     *
     * @param addTo The stack to have the heatmap's stack added onto it.
     * @param add The heatmap whose stack is to be appended.
     * @param min The minimum value in the stack.
     * @param max The maximum value in the stack.
     * @param es Manages the cpu threads.
     */
    public static void appendHM(MyImageStack addTo, HeatMapCreator add, float min, float max, ExecutorService es) {

        addTo.concat(add.getStack(min, max, es));

    }

    /**
     * Presents the images. This method handles displaying image-based outputs
     * or saving them to file based on the {@code save} parameter. Raw vector
     * data saving (if enabled via UserInput) is handled elsewhere and is not
     * controlled by this method.
     *
     * @param vf The vector field.
     * @param coh The coherence.
     * @param az The azimuthal angles.
     * @param zen The zenith angles.
     * @param save Specifies how to present/save image outputs (fiji, tiff,
     * png).
     * @param ui The user input.
     * @param dims The dimensions.
     * @param vecImgDepth The depth of the vector image.
     * @param myImg The image worked on.
     */
    private void processResults(Save save, UsrInput ui, int vecImgDepth) {

        if (ui.heatMap) {
            present(az.getImagePlus("Azimuthal Angles", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Azimuthal");
            if (ui.dim.hasDepth())
                present(zen.getImagePlus("Zenith Angles", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Zenith");
        }
        if (ui.vectorField.is())
            present(
                    !ui.dim.hasDepth() && ui.overlay.orElse(false)
                    ? new MyImagePlus("Overlaid Nematic Vectors", ui.img.getImageStack(), ui.dim.depth).overlay(vf, Color.GREEN)
                    : vf.getImagePlus("Nematic Vectors", vecImgDepth),
                    save,
                    "N_PIG_images" + File.separatorChar + "vectors"
            );

        if (ui.coherence)
            present(coh.getImagePlus("Coherence", ui.dim.depth), save, "N_PIG_images" + File.separatorChar + "Coherence");
    }

    /**
     * Presents the image, either by showing it on Fiji or saving it as a file.
     * This method is specifically for ImagePlus outputs (TIFF, PNG, Fiji
     * display).
     *
     * @param image The image to be presented.
     * @param saveTo The desired save format (fiji, tiff, png).
     * @param filePath The base file path for saving (ignored if displaying in
     * Fiji).
     */
    private static void present(MyImagePlus image, Save saveTo, String filePath) {
        if (saveTo == Save.fiji) {

            image.setOpenAsHyperStack(true);

            image.show();

        } else if (saveTo == Save.tiff || saveTo == Save.png) {
            try {
                Files.createDirectories(Paths.get(filePath));
            } catch (IOException e) {
                System.err.println("Failed to create directory: " + filePath + " - " + e.getMessage());
            }
            image.saveSlices(filePath, saveTo == Save.tiff);
        }

    }

    @Override
    public void run() {
        System.out.println("fijiPlugin.FijiPlugin.run() Thread launched.");
        if (!ui.validParamaters())
            throw new RuntimeException("fijiPlugin.FijiPlugin.run() Invalid Parameters!");

        try {

            int vecImgDepth = 0;

            int framesPerIteration = framesPerRun();

            long startTime = System.currentTimeMillis();

            for (int i = 0; i < ui.img.getNFrames(); i += framesPerIteration)
                try (Handle handle = new Handle(); NeighborhoodPIG np = new NeighborhoodPIG(handle, ui.img.subset(i, framesPerIteration), ui)) {
                vecImgDepth = processNPIGResults(ui, handle, np);
            }

            awaitThreadTermination();
            printTime(startTime);

            processResults(save, ui, vecImgDepth);
        } catch (Exception ex) {
            System.out.println("fijiPlugin.FijiPlugin.run() " + ui.toString() + " " + ex.toString());
            Kernel.closeModule();
            throw ex;
        }
        if (!Array.allocatedArrays.isEmpty())
            throw new RuntimeException("Neighborhood PIG has a GPU memory leak.");
        Kernel.closeModule();
    }

}
