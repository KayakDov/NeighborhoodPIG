package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.DArray1d;
import JCudaWrapper.array.DArray2d;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.array.DStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {
        
        double[] cpuArray = new double[16];
        Arrays.setAll(cpuArray, i -> i*2);

        try (Handle handle = new Handle(); 
                DStrideArray3d array = new DStrideArray3d(2, 2, 2, 2).set(handle, cpuArray)) {

                System.out.println("main.Test.main() size = " + array.size());
                System.out.println("toString:\n" + array.toString());
                System.out.println("toString:\n" + Arrays.toString(array.get(handle)));
            
        }



    }
}
