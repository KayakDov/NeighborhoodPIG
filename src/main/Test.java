package main;

import JCudaWrapper.array.Double.DStrideArray3d;
import JCudaWrapper.array.Float.FArray1d;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    public static void main(String[] args) {
        
        try(Handle hand = new Handle(); FStrideArray3d gpu = new FStrideArray3d(3, 3, 3, 2)){
            float[] cpu = new float[3*3*3*2];
            for(int i = 0; i < cpu.length; i++) cpu[i] = i;
            gpu.set(hand, cpu);
            
            System.out.println("lines per layer = " + gpu.linesPerLayer());
            
            System.out.println(gpu.toString());
            
            FStrideArray3d sub = new FStrideArray3d(gpu, 0, 2, 0, 2);
            System.out.println("sub\n" + sub);
        }

    }
}
