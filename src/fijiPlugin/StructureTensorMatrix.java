package fijiPlugin;

import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public class StructureTensorMatrix implements AutoCloseable {

    private final Eigan eigen;
    private final DStrideArray3d orientationXY, orientationYZ, coherence;
    private final Handle handle;

    /**
     * Finds the structure tensor for every pixel in the image and stores them
     * in a column major format.
     *
     * @param handle The context.
     * @param grad The pixel intensity gradient of the image.
     * @param neighborhoodRad A square window considered a neighborhood around a
     * point. This is the distance from the center of the square to the nearest
     * point on the edge.
     * @param tolerance How close a number be to 0 to be considered 0.
     */
    public StructureTensorMatrix(Handle handle, Gradient grad, int neighborhoodRad, double tolerance) {

        this.handle = handle;
        
        eigen = new Eigan(neighborhoodRad, handle, tolerance);

        try (NeighborhoodProductSums nps = new NeighborhoodProductSums(handle, neighborhoodRad, grad.x)) {
            nps.set(grad.x, grad.x, eigen.depth(0, 0));
            nps.set(grad.x, grad.y, eigen.depth(0, 1));
            nps.set(grad.y, grad.y, eigen.depth(1, 1));
            nps.set(grad.x, grad.z, eigen.depth(0, 2));
            nps.set(grad.y, grad.z, eigen.depth(1, 2));
            nps.set(grad.z, grad.z, eigen.depth(2, 2));
        }

        eigen.copyLowerTriangleToUpper();

        eigen.setEigenVals().setEiganVectors();
        
        
//        int problemMat = eigen.vectors.firstTensorIndexOfNaN();//TODO: delete this
//        if(problemMat == -1) System.out.println("clean eigenvectors");
//        System.out.println("fijiPlugin.StructureTensorMatrix.<init>() matrix index " + problemMat);
//        System.out.println("fijiPlugin.StructureTensorMatrix.<init>() matrix\n" + strctTensors.getTensor(problemMat));
//        System.out.println("fijiPlugin.StructureTensorMatrix.<init>() values\n" + eigen.values.getTensor(problemMat));
//        System.out.println("fijiPlugin.StructureTensorMatrix.<init>() vectors\n" + eigen.vectors.getTensor(problemMat));


        orientationXY = grad.x.copyDim();
        orientationYZ = grad.x.copyDim();
        coherence = grad.x.copyDim();

        setVecs0ToPi();
        setCoherence(tolerance);
        setOrientations();
    }
//
//    /**
//     * Gets the structure tensor from pixel at the given row and column of the
//     * picture.
//     *
//     * @param row The row of the desired pixel.
//     * @param col The column of the desired pixel.
//     * @param layer The desired layer of the tensor.
//     * @param frame The desired frame of the tensor.
//     * @return The structure tensor for the given row and column.
//     */
//    public Matrix getTensor(int row, int col, int layer, int frame) {
//
//        return strctTensors.getMatrix(index(row, col) + layer*orientationXY.layerDist + frame*orientationXY.strideSize);
//    }

    /**
     * All the eigen vectors with y less than 0 are mulitplied by -1.
     *
     * @return The eigenvectors.
     */
    public final DArray3d setVecs0ToPi() {
        DArray3d eVecs = eigen.vectors;
        Kernel.run("vecToNematic", handle,
                eVecs.numLayers(),
                eVecs,
                P.to(eVecs.ld()),
                P.to(eVecs),
                P.to(eVecs.ld())
        );
        return eVecs;
    }

    /**
     * Sets the orientations from the eigenvectors.
     *
     */
    public final void setOrientations() {

        try (Kernel atan2 = new Kernel("atan2")) {

            int eiganVecLayerStride = eigen.vectors.ld()*eigen.vectors.linesPerLayer();
            atan2.run(handle,
                    orientationXY.size(),
                    eigen.vectors,
                    P.to(eiganVecLayerStride),
                    P.to(orientationXY),
                    P.to(1)
            );
            atan2.run(handle,
                    orientationXY.size(),
                    eigen.vectors,
                    P.to(eiganVecLayerStride),
                    P.to(orientationYZ),
                    P.to(1)
            );
        }
        
        
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
                P.to(eigen.values.entriesPerLine()),
                P.to(coherence),
                P.to(eigen.values.entriesPerLine() == 3),
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
    public DStrideArray3d getOrientationXY() {
        return orientationXY;
    }

    /**
     * Gets the matrix of orientations.
     *
     * @return Thew matrix of orientations.
     */
    public DStrideArray3d getOrientationYZ() {
        return orientationYZ;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void close() {
        
        eigen.close();
        orientationXY.close();
        orientationYZ.close();
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
