package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {

        try (
                Handle hand = new Handle(); 
                DArray2d a0 = new DArray2d(2, 2); 
                DArray2d a1 = new DArray2d(2, 2); 
                DArray2d a2 = new DArray2d(2, 2);
                DArray2d a3 = new DArray2d(2, 2);
                PArray2dToD2d p = new PArray2dToD2d(2, 2, 2, 2);
                ) {

            a0.set(hand, 1, 2, 3, 4);
            a1.set(hand, 4, 3, 2, 1);
            a2.set(hand, 9, 12, 3, -1);
            a3.set(hand, 7, 8, 70, 80);
            
            p.set(hand, a0, a1, a2, a3);
            

            System.out.println(Arrays.toString(p.get(hand)));

        }

    }

    /**
     * Checks if all the elements of the array are finite. The check is
     * performed on the cpu.  It is for testing purposes only.
     *
     * @param array The array to be checked.
     * @param handle
     * @return true if all the elements are finite. False otherwise.
     */
    public static int nonFiniteCount(FArray array, Handle handle) {
        int count = 0;
        float[] cpuArray = array.get(handle);
        for (int i = 0; i < cpuArray.length; i++)
            if (!Float.isFinite(cpuArray[i])) count++;
        return count;
    }
}
