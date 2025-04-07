package main;

import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.resourceManagement.Handle;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {
        
        try(Handle hand = new Handle(); FArray2d gpu = new FArray2d(3, 3); ){
            
            float[] cpu = new float[]{1,2,3,  3,2,1,  4,5,6};
            
            gpu.set(hand, cpu);
     
        }

    }
}
