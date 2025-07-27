package imageWork;

import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
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
    private final P2dToF2d gpuVectorData, coherence;
    private final Handle handle;
    private final Path dstDir;
    private final int scaleXY, scaleZ;
    private final double tolerance;

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
     * @param coherence The coherence of the image.
     * @param tolerance Vectors with a  coherence below this threshold will not be printed.
     */
    public DatSaver(Dimensions dim, P2dToF2d gpuVectorData, Handle handle, Path dstDirectory, int scaleXY, int scaleZ, P2dToF2d coherence, double tolerance) {
        this.dim = dim;
        this.gpuVectorData = gpuVectorData;
        this.handle = handle;
        this.dstDir = dstDirectory;
        this.scaleXY = scaleXY;
        this.scaleZ = scaleZ;
        this.coherence = coherence;
        this.tolerance = tolerance;

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

            for (int t = 0; t < dim.batchSize; t++)
                saveFrame(t);

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
        try (PrintWriter vecWriter = new PrintWriter(Paths.get(dstDir.toString(), "vecFrame_" + t + ".dat").toString()); PrintWriter cohWriter = new PrintWriter(Paths.get(dstDir.toString(), "coherenceFrame_" + t + ".dat").toString());) {

            if (dim.hasDepth())
                vecWriter.println("# x y z nx ny nz");
            else
                vecWriter.println("# x y nx ny");
            cohWriter.println("#coherence vectorType");

            VecManager2d vecSlice = new VecManager2d(dim);
            float[] coherenceSlice = new float[dim.layerSize()];

            for (int z = 0; z < dim.depth; z++)
                appendSliceToFile(vecWriter, vecSlice.setFrom(gpuVectorData, t, z, handle), coherenceSlice, z);

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
     * @param vecWrite The writer.vecLayer@param vm Manages accessing the
     * vector.
     * @param vecLayer The current layer of vectors.
     * @param coherenceSlice The current coherence.
     * @param sliceIndex The index of the current Z-slice.
     */
    public void appendSliceToFile(PrintWriter vecWrite, VecManager2d vecLayer, float[] coherenceSlice, int sliceIndex) {

        Point3d vec = new Point3d();

        for (int col = 0; col < dim.width; col++) {
            int colInd = col * dim.height;

            for (int row = 0; row < dim.height; row++) {

                vecLayer.get(row, col, vec);
                float coh = coherenceSlice[colInd + row];

                if (coh > tolerance)
                    if (vec.isFinite())
                        if (dim.hasDepth())
                            vecWrite.printf("%d %d %d %f %f %f%n", col * scaleXY, row * scaleXY, sliceIndex * scaleZ, vec.x(), vec.y(), vec.z());
                        else
                            vecWrite.printf("%d %d %f %f%n", col * scaleXY, row * scaleXY, vec.x(), vec.y());

            }
        }

    }

}
