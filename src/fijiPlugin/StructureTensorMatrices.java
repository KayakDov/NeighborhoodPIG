package fijiPlugin;

import FijiInput.UserInput;
import JCudaWrapper.array.Float.FArray3d;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import imageWork.VecManager;
import java.util.Arrays;
import java.util.stream.IntStream;
import main.Test;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrices implements AutoCloseable {

    public final Eigen eigen;
    private final FStrideArray3d azimuth, zenith, coherence;
    private final Handle handle;

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

        this.handle = handle;

        float bigTolerance = 255*255*ui.neighborhoodSize.xyR*ui.neighborhoodSize.xyR*ui.neighborhoodSize.zR*1e-7f;
        
        eigen = new Eigen(handle, grad, ui.downSampleFactorXY, bigTolerance);

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, ui.neighborhoodSize, grad.x[0])) {
            for (int i = 0; i < 3; i++)
                for (int j = i; j < 3; j++)
                    nps.set(grad.x[i], grad.x[j], eigen.at(i, j));
        }
       
        eigen.set(Math.min(grad.depth, 2));
                
        azimuth = new FStrideArray3d(grad.height / ui.downSampleFactorXY, grad.width / ui.downSampleFactorXY, grad.depth, grad.batchSize);
        
        zenith = azimuth.copyDim();
        coherence = azimuth.copyDim();
        
        setVecs0ToPi();        
                
        setCoherence(bigTolerance);

//        System.out.println("fijiPlugin.StructureTensorMatrices.<init>() vector att (800, 575) is " 
//                + Arrays.toString(new VecManager(grad).setFrom(eigen.vectors, 0, handle).get(575, 800, 0)) + 
//                " has index " + new VecManager(grad).setFrom(eigen.vectors, 0, handle).vecIndex(575, 800, 0)/3);
//        
        
        setOrientations(1e-6f);
        
//        System.out.println("\nfijiPlugin.StructureTensorMatrices.<init>() azimuth angle is: " + azimuth.getAt(575, 800).getf(handle));
        
        
    }

    /**
     * All the eigen vectors with y less than 0 are mulitplied by -1.
     *
     */
    public final void setVecs0ToPi() {
        FArray3d eVecs = eigen.vectors;
        Kernel.run("vecToNematic", handle,
                coherence.size(),
                eVecs,
                P.to(eVecs.ld()),
                P.to(eVecs.entriesPerLine()),
                P.to(eVecs),
                P.to(eVecs.ld()),
                P.to(eVecs.entriesPerLine())
        );
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     * @param tolerance What is considered 0.
     */
    public final void setOrientations(float tolerance) {

        Kernel.run("toSpherical", handle, azimuth.size(),
                eigen.vectors,
                P.to(eigen.vectors.entriesPerLine()),
                P.to(eigen.vectors.ld()),
                P.to(azimuth),
                P.to(azimuth.entriesPerLine()),
                P.to(azimuth.ld()),
                P.to(zenith),
                P.to(zenith.entriesPerLine()),
                P.to(zenith.ld()),
                P.to(0.01f)
        );

    }

    /**
     * Sets and returns the coherence matrix.
     *
     * @param tolerance Numbers closer to 0 than may be considered 0.
     * @return The coherence matrix.
     */
    public final FStrideArray3d setCoherence(float tolerance) {
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
    public FStrideArray3d getCoherence() {
        return coherence;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public FStrideArray3d azimuthAngle() {
        return azimuth;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public FStrideArray3d zenithAngle() {
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
