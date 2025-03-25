package main;

import JCudaWrapper.array.Double.DStrideArray3d;
import JCudaWrapper.array.Float.FArray1d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {
        
        try(Handle hand = new Handle(); FArray2d d2= new FArray2d(2, 2); FArray1d d1 = new FArray1d(4)){
            d2.set(hand, 1, 2, 3, 4);
            System.out.println(d2.toString());
            d2.get(hand, d1);
            System.out.println(d1.toString());
        }

    }
}
