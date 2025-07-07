package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import MathSupport.Point3d;
import fijiPlugin.Dimensions;
import java.io.FileNotFoundException;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * A class to orchestrate the saving of vector data for all slices and frames of
 * an image, performing GPU reads sequentially and file saving in parallel.
 *
 * It utilizes the provided VecManager to handle per-slice vector data and file
 * writing.
 */
public class DatSaver {

    private final Dimensions dim;
    private final PArray2dToF2d gpuVectorData;
    private final Handle handle;
    private final Path dstDir;
    private final int scaleXY, scaleZ;

    /**
     * Constructs a new VectorDataSaver.
     *
     * @param dim The overall dimensions of the image stack (height, width,
     * depth, nFrames, numComponentsPerPixel). This object should represent the
     * full dimensions of the original image, as VecManager's internal 'dim'
     * uses this to determine 2D/3D output format.
     * @param gpuVectorData The GPU-resident vector data source (e.g., from
     * NeighborhoodPIG.getEigenVectors()).
     * @param handle The JCuda handle for memory operations.
     * @param dstDirectory The base directory where the output files will be
     * saved.
     * @param scaleXY Scale the vector xy locations.
     * @param scaleZ Scale the vector z locations.
     */
    public DatSaver(Dimensions dim, PArray2dToF2d gpuVectorData, Handle handle, Path dstDirectory, int scaleXY, int scaleZ) {
        this.dim = dim;
        this.gpuVectorData = gpuVectorData;
        this.handle = handle;
        this.dstDir = dstDirectory;
        this.scaleXY = scaleXY;
        this.scaleZ = scaleZ;

    }

    /**
     * Saves all slices in all frames to separate files. GPU data reading
     * (transfer from GPU to CPU memory) is done sequentially for each slice,
     * while the subsequent file writing operation for that slice is
     * parallelized.
     *
     */
    public void saveAllVectors() {

        try {
            Files.createDirectories(dstDir);


            for (int t = 0; t < dim.batchSize; t++) saveFrame(t);

        } catch (IOException ex) {
            Logger.getLogger(DatSaver.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    /**
     * Saves a frame.
     *
     * @param t The index of the frame.
     * @throws FileNotFoundException
     */
    private void saveFrame(int t) {
        try (PrintWriter pw = new PrintWriter(Paths.get(dstDir.toString(), "frame_" + t + ".dat").toString())) {

            if (dim.hasDepth()) pw.println("# x y z nx ny nz");
            else pw.println("# x y nx ny");

            VecManager2d vm = new VecManager2d(dim);

            for (int z = 0; z < dim.depth; z++)
                appendSliceToFile(pw, vm.setFrom(gpuVectorData, t, z, handle), z);
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DatSaver.class.getName()).log(Level.SEVERE, null, ex);
        }

    }


    /**
     * Saves all vectors from the current slice managed by this `VecManager` to
     * a text file. Each line in the file will represent a pixel/voxel and
     * contain its spatial coordinates (x, y, z) followed by its primary
     * orientation vector components (nx, ny, nz).
     *
     * The file will be named in the format
     * `vectors_f<frameIndex>_s<sliceIndex>.txt` within the specified output
     * directory.
     *
     * @param writer The writer.
     * @param vm Manages accessing the vector.
     * @param sliceIndex The index of the current Z-slice.
     */
    public void appendSliceToFile(PrintWriter writer, VecManager2d vm, int sliceIndex) {

        Point3d vec = new Point3d();

        for (int row = 0; row < dim.height; row++)
            for (int col = 0; col < dim.width; col++) {

                vm.get(row, col, vec);
                if (dim.hasDepth()) writer.printf("%d %d %d %f %f %f%n", col * scaleXY, row * scaleXY, sliceIndex * scaleZ, vec.getX(), vec.getY(), vec.getZ());
                else writer.printf("%d %d %f %f%n", col*scaleXY, row * scaleXY, vec.getX(), vec.getY());
            }

    }

}
