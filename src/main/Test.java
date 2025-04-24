package main;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.resourceManagement.Handle;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {

        try (Handle hand = new Handle(); FArray2d gpu = new FArray2d(3, 3);) {

            float[] cpu = new float[]{1, 2, 3, 3, 2, 1, 4, 5, 6};

            gpu.set(hand, cpu);

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
