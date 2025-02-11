package fijiPlugin;

import JCudaWrapper.array.DArray1d;
import JCudaWrapper.array.DArray2d;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.IArray1d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;

/**
 * A set of 3x3 matrices, their eigenvectors and values.
 *
 * @author E. Dov Neimand
 */
public class Eigan {

    private final Handle handle;
    private final double tolerance;

    /**
     * Gets a depth (z dimensional column) of the
     */
    public interface N2To1dArrayBuilder {

        public DArray1d get(int row, int col);
    }

    /**
     * Each layer in this matrix is for a different pixel, in column major
     * order.
     */
    public final DArray3d structureTensors;

    /**
     * Each row of this matrix is for the tensor of a different pixel.
     */
    public final DArray2d values;

    /**
     * Each layer is the eiganvectors for the corresponding tensor.
     */
    private DArray3d vectors;

    public Eigan(int numPixels, Handle handle, N2To1dArrayBuilder array, double tolerance) {
        this.handle = handle;
        this.tolerance = tolerance;
        structureTensors = new DArray3d(3, 3, numPixels);

        for (int i = 0; i < 3; i++)
            for (int j = 0; j <= i; j++) {
                DArray1d a = array.get(i, j);
                a.get(handle, structureTensors.depth(i, j));
                if (i != j) a.get(handle, structureTensors.depth(j, i));
            }
        values = new DArray2d(numPixels, 3);
        setEigenVals();
        vectors = new DArray3d(3, 3, numPixels);
        setEiganVectors();

    }

    /**
     * Sets the eiganvalues. *
     */
    public final void setEigenVals() {
        Kernel.run("eigenValsBatch", handle,
                structureTensors.numLayers(), structureTensors, P.to(values), P.to(tolerance));
    }

    /**
     * Sets the eiganvectors.
     */
    public final void setEiganVectors() {//TODO: add ld

        try (IArray1d pivotFlags = new IArray1d(structureTensors.numLayers());
                DArray3d workSpace = new DArray3d(3, 3, structureTensors.numLayers())) {

            for (int i = 0; i < 3; i++) 

                Kernel.run("eigenVecBatch", handle,
                        structureTensors.numLayers(),
                        workSpace.set(handle, structureTensors),
                        P.to(3),
                        P.to(3),
                        P.to(workSpace.ld()),
                        P.to(vectors),
                        P.to(vectors.ld()),
                        P.to(i),
                        P.to(values),
                        P.to(pivotFlags),
                        P.to(tolerance)
                );

            

        }
    }
}
