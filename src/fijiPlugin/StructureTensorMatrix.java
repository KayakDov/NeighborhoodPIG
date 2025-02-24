package fijiPlugin;

import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrix implements AutoCloseable {

    private final Eigan eigen;
    private final DStrideArray3d azimuthalAngles, polarAngles, coherence;
    private final Handle handle;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param handle The context.
     * @param grad The pixel intensity gradient of the image.
     * @param nRad A square window considered a neighborhood around a
     * point. This is the distance from the center of the square to the nearest
     * point on the edge.
     * @param tolerance How close a number be to 0 to be considered 0.
     */
    public StructureTensorMatrix(Handle handle, Gradient grad, NeighborhoodDim nRad, double tolerance) {

        this.handle = handle;

        eigen = new Eigan(handle, grad, tolerance);

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, nRad, grad.x[0])) {
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    nps.set(grad.x[i], grad.x[j], eigen.at(i, j));
        }

        eigen.setEigenVals().setEiganVectors();

        azimuthalAngles = grad.copyDim();
        polarAngles = grad.copyDim();
        coherence = grad.copyDim();

        setVecs0ToPi();
        setCoherence(tolerance);
        setOrientations(tolerance);
    }

    /**
     * All the eigen vectors with y less than 0 are mulitplied by -1.
     *
     * @return The eigenvectors.
     */
    public final DArray3d setVecs0ToPi() {
        DArray3d eVecs = eigen.vectors1;
        Kernel.run("vecToNematic", handle,
                eigen.size(),
                eVecs,
                P.to(eVecs.ld()),
                P.to(eVecs.entriesPerLine()),
                P.to(eVecs),
                P.to(eVecs.ld()),
                P.to(eVecs.entriesPerLine())
        );
        return eVecs;
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     * @param tolerance What is considered 0.
     */
    public final void setOrientations(double tolerance) {

        Kernel.run("toSpherical",  handle, azimuthalAngles.size(),
                
                eigen.vectors1,                
                P.to(eigen.vectors1.entriesPerLine()),
                P.to(eigen.vectors1.ld()),
                
                P.to(azimuthalAngles),
                P.to(azimuthalAngles.entriesPerLine()),
                P.to(azimuthalAngles.ld()),
                
                P.to(polarAngles),
                P.to(polarAngles.entriesPerLine()),
                P.to(polarAngles.ld()),
                
                P.to(.01)
        );

    }

    /**
     * Sets and returns the coherence matrix.
     *
     * @param tolerance Numbers closer to 0 than may be considered 0.
     * @return The coherence matrix.
     */
    public final DStrideArray3d setCoherence(double tolerance) {
        Kernel.run("coherence", handle,
                coherence.size(),
                eigen.values,
                P.to(eigen.values.ld()),
                P.to(eigen.values.entriesPerLine()),
                P.to(coherence),
                P.to(coherence.ld()),
                P.to(coherence.entriesPerLine()),
                P.to(tolerance)
        );

        return coherence;
    }

    /**
     * The coherence matrix.
     *
     * @return The coherence matrix.
     */
    public DStrideArray3d getCoherence() {
        return coherence;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public DStrideArray3d getAzimuthalAngles() {
        return azimuthalAngles;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public DStrideArray3d getPolarAngles() {
        return polarAngles;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {

        eigen.close();
        azimuthalAngles.close();
        polarAngles.close();
        coherence.close();
    }

    /**
     * The eigenvalues and vectors of the structure tensors.
     *
     * @return The eigenvalues and vectors of the structure tensors.
     */
    public Eigan getEigen() {
        return eigen;
    }

}
