package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;
import JCudaWrapper.resourceManagement.Handle;



/**
 * The eigen values and vectors for a 3x3 batch matrix.
 *
 * @author E. Dov Neimand
 */
public class Eigen implements AutoCloseable {

    public VectorsStride values;
    public MatricesStride vectors;

    /**
     * Computes the eigenvalues and vectors of the matrices.
     * @param mats A set of 2x2 or 3x3 matrices.
     * 
     */
    public Eigen(MatricesStride mats, double tolerance) {
        try (DArray workSpace = DArray.empty(mats.dArray().length)) {

            switch (mats.height) {
                case 2: values = mats.computeVals2x2(new Vector(mats.getHandle(), workSpace.subArray(0, mats.getBatchSize()), 1)); break;
                case 3: values = mats.computeVals3x3(new Vector(mats.getHandle(), workSpace.subArray(0, mats.width* mats.getBatchSize()), 1)); break;
                default: throw new UnsupportedOperationException("Currently the Eigen method only works for 2x2 and 3x3 matrices.  Your matrix is " + mats.height + "x" + mats.width); 
            }
            
            vectors =  mats.computeVecs(values, workSpace, tolerance);

        }

    }
    
    public static void main(String[] args) {
        try(Handle handle = new Handle(); DArray array = new DArray(handle, 1,2,2,3,4,5,5,6);){
            MatricesStride mst = new MatricesStride(handle, array, 2, 2, 2, 4, 2);
            Matrix m1 = new Matrix(handle, array, 2, 2);
            Matrix m2 = new Matrix(handle, array.subArray(4), 2, 2);
            m1.power(2);
            m2.power(2);
            
            
            System.out.println(mst.toString());
            try(Eigen eig = new Eigen(mst, 1e-13)){
                System.out.println("values \n" + eig.values.toString());
                System.out.println("vectors \n" + eig.vectors.toString());
            }
           
            
        }
    }

    @Override
    public void close(){
        values.close();
        vectors.close();
    }

}
