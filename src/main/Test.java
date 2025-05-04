package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Pointer.to2d.PointerArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;

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
                PointerArray2dToD2d p = new PointerArray2dToD2d(2, 2, 2, 2);
                ) {

            a0.set(hand, 1, 2, 3, 4);
            a1.set(hand, 4, 3, 2, 1);
            a2.set(hand, 9, 9, 3, 3);
            a3.set(hand, 7, 8, 7, 8);
            
            p.set(hand, a0, a1, a2, a3);
            
            System.out.println(p.get(2).getVal(hand).toString());

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
