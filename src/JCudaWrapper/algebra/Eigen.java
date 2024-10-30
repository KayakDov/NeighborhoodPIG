package JCudaWrapper.algebra;

import JCudaWrapper.array.DArray;



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

    @Override
    public void close(){
        values.close();
        vectors.close();
    }

}
