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
public class Eigan implements AutoCloseable{//TODO fix spelling to eigen

    private final Handle handle;
    private final double tolerance;

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
    public final DArray3d vectors;

    public Eigan(int numPixels, Handle handle, double tolerance) {
        this.handle = handle;
        this.tolerance = tolerance;
        structureTensors = new DArray3d(3, 3, numPixels);

        values = new DArray2d(3, numPixels);
        vectors = new DArray3d(3, 3, numPixels);

    }

    /**
     * The depth vector.
     * @param row The row of the desired vector.
     * @param col The column of the desired vector.
     * @return The depth vector.
     */
    public DArray1d depth(int row, int col){
        return structureTensors.depth(row, col);
    }
    
    /**
     * Copies the lower triangle to the upper one.
     */
    public void copyLowerTriangleToUpper(){
        depth(1, 0).set(handle, depth(0, 1));
        depth(2, 0).set(handle, depth(0, 2));
        depth(2, 1).set(handle, depth(1, 2));
    }
    
    /**
     * Sets the eiganvalues. *
     * @return this
     */
    public final Eigan setEigenVals() {
        Kernel.run("eigenValsBatch", handle,
                structureTensors.layersPerGrid(), 
                structureTensors,
                P.to(structureTensors.ld()),
                P.to(values),
                P.to(values.ld()),
                P.to(tolerance)
        );
        return this;
    }

    /**
     * Sets the eiganvectors.
     * @return this
     */
    public final Eigan setEiganVectors() {

        try (IArray1d pivotFlags = new IArray1d(structureTensors.layersPerGrid());
                DArray3d workSpace = new DArray3d(3, 3, structureTensors.layersPerGrid())) {

            for (int i = 0; i < 3; i++) 

                Kernel.run("eigenVecBatch", handle,
                        structureTensors.layersPerGrid(),//TODO: some vectors are getting infinity!
                        workSpace.set(handle, structureTensors),
                        P.to(3),
                        P.to(3),
                        P.to(workSpace.ld()),
                        P.to(vectors),
                        P.to(vectors.ld()),
                        P.to(i),
                        P.to(values),
                        P.to(values.ld()),
                        P.to(pivotFlags),
                        P.to(tolerance)
                );
        }        
        
        return this;
    }
    
    
    /**
     * {@inheritDoc }
     */
    @Override
    public void close() {
        structureTensors.close();
        values.close();
        vectors.close();
    }
}
