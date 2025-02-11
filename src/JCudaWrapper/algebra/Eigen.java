package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.IArray;
import JCudaWrapper.resourceManagement.Handle;

/**
 * The eigenvalues and vectors for a 3x3 batch matrix.
 *
 * @author E. Dov Neimand
 */
public class Eigen implements AutoCloseable {

    public VectorsStride values;
    public MatricesStride vectors;

    /**
     * Computes the eigenvalues and vectors of the matrices.
     *
     * @param mats A set of 2x2 or 3x3 matrices.
     * @param tolerance Used to say values are essentially 0.
     *
     */
    public Eigen(MatricesStride mats, double tolerance) {

        try (DArray3d workSpaceD = new DArray3d(mats.width * mats.array().size());
                IArray workSpaceI = new IArray(mats.width * mats.width * mats.batchSize)) {

            values = mats.height == 2
                    ? mats.computeVals2x2(new Vector(mats.getHandle(), workSpaceD.sub(0, mats.getBatchSize()), 1))
                    : mats.computeVals3x3(tolerance);

            vectors = mats.computeVecs(values, workSpaceD, workSpaceI, tolerance);            

        }

        
    }
    
    public static void main(String[] args) {
        try (Handle handle = new Handle();
                DArray3d array = new DArray3d(handle, 6, 0, 0,   0, 0, 0,   0, 0, 0, 1, 0, 1, 0, 2, 1, 1, 1, 3);) {

            MatricesStride mst = new MatricesStride(handle, array, 3, 3, 3, 9, 1);

            mst.getMatrix(0).power(2);
//            mst.getMatrix(1).power(2);

            System.out.println(mst.toString());

            try (Eigen eig = new Eigen(mst, 1e-12)) {
                System.out.println("values \n" + eig.values.toString() + "\n\n");
                System.out.println("vectors \n" + eig.vectors.toString());
            }
            

        }
    }

    @Override
    public String toString() {
        return "values \n" + values.toString() + "\nvectors \n" + vectors.toString();
    }

    @Override
    public void close() {
        values.close();
        vectors.close();
    }

}
