package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.PArray;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {

        test2dto2d();
    }

    /**
     * Checks if all the elements of the array are finite. The check is
     * performed on the cpu. It is for testing purposes only.
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
    
    public static void test2dto2d(){
                try (Handle hand = new Handle(); PArray2dToD2d p = new PArray2dToD2d(2, 2, 2, 2);) {
            

            DArray2d[] a = new DArray2d[]{
                new DArray2d(2, 2),
                new DArray2d(2, 2),
                new DArray2d(2, 2),
                new DArray2d(2, 2)
            };

            a[0].set(hand, 1, 2, 3, 4);
            a[1].set(hand, 4, 3, 2, 1);
            a[2].set(hand, 9, 12, 3, -1);
            a[3].set(hand, 7, 8, 70, 80);

            p.set(hand, a);
            p.get(1, 0).set(hand, a[3]);

            System.out.println(p.toString());

        }

    }
}
//verena
