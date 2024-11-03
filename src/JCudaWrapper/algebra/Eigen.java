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
     * Computes the eigen values and vectors of the matrices.
     * @param mats A set of 2x2 or 3x3 matrices.
     * @param is3x3 is it 2 or 3
     */
    public Eigen(MatricesStride mats, boolean is3x3) {
        try (DArray workSpace = DArray.empty(mats.dArray().length)) {

            if (!is3x3) 
                values = mats.computeVals2x2(new Vector(mats.getHandle(), workSpace.subArray(0, mats.getBatchSize()), 1));
            else values = mats.computeVals3x3(new Vector(mats.getHandle(), workSpace.subArray(0, mats.width), 1));
            
            vectors =  mats.computeVecs(values, workSpace);

        }

    }
    
    public static void main(String[] args) {
        try(Handle handle = new Handle(); DArray array = new DArray(handle, 1,2,2,3,5,6,6,7);){
            MatricesStride mst = new MatricesStride(handle, array, 2, 2, 2, 4, 2);
            Matrix m1 = new Matrix(handle, array, 2, 2);
            Matrix m2 = new Matrix(handle, array.subArray(4), 2, 2);
            m1.power(2);
            m2.power(2);
            System.out.println(mst.toString());
            try(Eigen eig = new Eigen(mst, false)){
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
