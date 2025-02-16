package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.DArray1d;
import JCudaWrapper.array.DArray2d;
import JCudaWrapper.array.DArray3d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {

        try (Handle handle = new Handle(); 
                DArray3d array = new DArray3d(2, 2, 2).set(handle, 1, 2, 3, 4, 5, 6, 7 ,8)) {

                System.out.println("toString:\n" + array.toString());
            
        }



    }
}
