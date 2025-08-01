package fijiPlugin;

import FijiInput.UsrInput;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import ij.ImagePlus;
import imageWork.MyImagePlus;
import main.Test;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrices implements AutoCloseable {

    public final Eigen eigen;
    public final P2dToF2d azimuth, zenith, coherence, vectors;//TODO:store these on cpu instead of gpu.
    public final Dimensions dim, downSampled;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param handle The context.
     * @param imp The image for which the structure tensors are to be generated.
     * @param ui User selected specifications.
     *
     */
    public StructureTensorMatrices(Handle handle, MyImagePlus imp, UsrInput ui) {
        try (Gradient grad = new Gradient(handle, imp, ui)) {
            dim = grad.dim;            
            
            
            
            eigen = new Eigen(handle, dim, ui);
            
            try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, ui.neighborhoodSize, dim)) {

                for (int i = 0; i < dim.num(); i++)
                    for (int j = i; j < dim.num(); j++)
                        nps.set(grad.x[i], grad.x[j], eigen.getMatValsAt(i, j));
            }
        }
        
        downSampled = dim.downSample(handle, ui.downSampleFactorXY, ui.downSampleFactorZ.orElse(1));
        
        azimuth = downSampled.emptyP2dToF2d(handle);
        zenith = dim.hasDepth() ? downSampled.emptyP2dToF2d(handle) : null;
        coherence = downSampled.emptyP2dToF2d(handle);
        vectors = new P2dToF2d(downSampled.depth, downSampled.batchSize, downSampled.height * dim.num(), downSampled.width, handle);
                
        eigen.set(dim.num() - 1, vectors, coherence, azimuth, zenith, downSampled).close();
        
//        System.out.println("fijiPlugin.StructureTensorMatrices.<init>()\n" + Test.format(vectors.toString()));
        
    }


    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {

        eigen.close();
        azimuth.close();
        if (zenith != null) zenith.close();
        coherence.close();
        vectors.close();
        dim.close();
        downSampled.close();
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
                + (dim.hasDepth() ? "\n\nxz\n" + eigen.getMatValsAt(0, 2).toString() : "")
                + "\n\nyy\n" + eigen.getMatValsAt(1, 1).toString()
                + (dim.hasDepth() ? "\n\nyz\n" + eigen.getMatValsAt(1, 2).toString()
                + "\n\nzz\n" + eigen.getMatValsAt(2, 2).toString() : "");
    }

    /**
     * The neighborhood gradient at each pixel, stored so that each column is
     * consecutive 3-dimensional vectors.
     *
     * @return The neighborhood gradient at each pixel, stored so that each
     * column is consecutive 3-dimensional vectors.
     */
    public P2dToF2d getVectors() {
        return vectors;
    }

}
