package fijiPlugin;

import FijiInput.UserInput;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrices extends Dimensions implements AutoCloseable {

    public final Eigen eigen;
    private final PArray2dToD2d azimuth, zenith, coherence;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param handle The context.
     * @param grad The pixel intensity gradient of the image.
     * @param ui User selected specifications.
     *
     */
    public StructureTensorMatrices(Handle handle, Gradient grad, UserInput ui) {
        super(handle, grad.x[0]);

        float bigTolerance = 255*255*ui.neighborhoodSize.xyR*ui.neighborhoodSize.xyR*ui.neighborhoodSize.zR*5e-7f;
        
        eigen = new Eigen(handle, grad, ui.downSampleFactorXY, bigTolerance);

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, ui.neighborhoodSize, grad.x[0])) {
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    nps.set(grad.x[i], grad.x[j], eigen.at(i, j));
        }
       
        azimuth = new PArray2dToD2d(depth, batchSize, height / ui.downSampleFactorXY, width / ui.downSampleFactorXY).initTargets(handle);
        zenith = azimuth.copyDim(handle);
        coherence = azimuth.copyDim(handle);        
        
        eigen.set(Math.min(grad.depth, 2), coherence, azimuth, zenith);//TODO: restore higher eigen index
    }


    /**
     * The coherence matrix.
     *
     * @return The coherence matrix.
     */
    public PArray2dToD2d getCoherence() {
        return coherence;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public PArray2dToD2d azimuthAngle() {
        return azimuth;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public PArray2dToD2d zenithAngle() {
        return zenith;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {

        eigen.close();
        azimuth.close();
        zenith.close();
        coherence.close();
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
        return "xx\n" + eigen.at(0, 0).toString()
                + "\n\nxy\n" + eigen.at(0, 1).toString()
                + "\n\nxz\n" + eigen.at(0, 2).toString()
                + "\n\nyy\n" + eigen.at(1, 1).toString()
                + "\n\nyz\n" + eigen.at(1, 2).toString()
                + "\n\nzz\n" + eigen.at(2, 2).toString();
    }

}


//
//    /**
//     * All the eigen vectors with y less than 0 are mulitplied by -1.
//     *
//     */
//    public final void setVecs0ToPi() {
//        FArray3d eVecs = eigen.vectors;
//        Kernel.run("vecToNematic", handle,
//                coherence.size(),
//                eVecs,
//                P.to(eVecs.ld()),
//                P.to(eVecs.entriesPerLine()),
//                P.to(eVecs),
//                P.to(eVecs.ld()),
//                P.to(eVecs.entriesPerLine())
//        );
//    }
//
//    /**
//     * Sets the orientations from the eigenvectors.
//     *
//     * @param tolerance What is considered 0.
//     */
//    public final void setOrientations(float tolerance) {
//
//        Kernel.run("toSpherical", handle, azimuth.size(),
//                eigen.vectors,
//                P.to(eigen.vectors.entriesPerLine()),
//                P.to(eigen.vectors.ld()),
//                P.to(azimuth),
//                P.to(azimuth.entriesPerLine()),
//                P.to(azimuth.ld()),
//                P.to(zenith),
//                P.to(zenith.entriesPerLine()),
//                P.to(zenith.ld()),
//                P.to(0.01f)
//        );



