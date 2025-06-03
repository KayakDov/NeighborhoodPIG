package fijiPlugin;

import FijiInput.UserInput;
import JCudaWrapper.array.Pointer.to2d.PArray2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrices implements AutoCloseable {

    public final Eigen eigen;
    private final PArray2dToF2d azimuth, zenith, coherence, vectors;//TODO:store these on cpu instead of gpu.
    public final Dimensions dim;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param handle The context.
     * @param imp The image for which the structure tensors are to be generated.
     * @param ui User selected specifications.
     *
     */
    public StructureTensorMatrices(Handle handle, ImagePlus imp, UserInput ui) {
        try (Gradient grad = new Gradient(handle, imp, ui)) {
            
            dim = grad.dim;

            float bigTolerance = 255 * 255 * ui.neighborhoodSize.xyR * ui.neighborhoodSize.xyR * ui.neighborhoodSize.zR * 5e-7f;

            eigen = new Eigen(handle, dim, ui.downSampleFactorXY, bigTolerance);

            try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, ui.neighborhoodSize, dim)) {

                int numDim = dim.hasDepth()? 3 : 2;
                for (int i = 0; i < numDim; i++)
                    for (int j = i; j < numDim; j++)
                        nps.set(grad.x[i], grad.x[j], eigen.getMatValsAt(i, j));
            }
        }

        try (Dimensions downSampled = new Dimensions(handle, dim.height / ui.downSampleFactorXY, dim.width / ui.downSampleFactorXY, dim.depth, dim.batchSize)) {  //TODO: make sure downsampled dimensions are used everywhere they are needed!

            
            azimuth = downSampled.emptyP2dToF2d(handle);
            zenith = dim.hasDepth()?downSampled.emptyP2dToF2d(handle):null;
            coherence = downSampled.emptyP2dToF2d(handle);
            vectors = new PArray2dToF2d(downSampled.depth, downSampled.batchSize, downSampled.height * (dim.hasDepth()?3:2), downSampled.width, handle);

            eigen.set(Math.min(dim.depth, 2), vectors, coherence, azimuth, zenith, downSampled.getGpuDim());//TODO: restore later eigenevec

            
            eigen.close();
        }
    }

    /**
     * The coherence matrix.
     *
     * @return The coherence matrix.
     */
    public PArray2dToF2d getCoherence() {
        return coherence;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public PArray2dToF2d azimuthAngle() {
        return azimuth;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public PArray2dToF2d zenithAngle() {
        return zenith;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {

        eigen.close();
        azimuth.close();
        if(zenith != null) zenith.close();
        coherence.close();
        vectors.close();
        dim.close();
    }

    /**
     * The eigenvalues and vectors of the structure tensors.
     *
     * @return The eigenvalues and vectors of the structure tensors.
     */
    public Eigen getEigen() {
        return eigen;
    }

    @Override
    public String toString() {
        return "xx\n" + eigen.getMatValsAt(0, 0).toString()
                + "\n\nxy\n" + eigen.getMatValsAt(0, 1).toString()
                + (dim.hasDepth()?"\n\nxz\n" + eigen.getMatValsAt(0, 2).toString():"")
                + "\n\nyy\n" + eigen.getMatValsAt(1, 1).toString()
                + (dim.hasDepth()?"\n\nyz\n" + eigen.getMatValsAt(1, 2).toString()
                + "\n\nzz\n" + eigen.getMatValsAt(2, 2).toString() : "");
    }

    /**
     * The neighborhood gradient at each pixel, stored so that each column is
     * consecutive 3-dimensional vectors.
     *
     * @return The neighborhood gradient at each pixel, stored so that each
     * column is consecutive 3-dimensional vectors.
     */
    public PArray2dToF2d getVectors() {
        return vectors;
    }

}
