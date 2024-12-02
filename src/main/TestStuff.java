package main;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.algebra.VectorsStride;
import JCudaWrapper.array.DArray;
import JCudaWrapper.array.DStrideArray;
import JCudaWrapper.array.IArray;
import JCudaWrapper.array.KernelManager;
import JCudaWrapper.resourceManagement.Handle;
import java.util.Arrays;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.runtime.cudaStream_t;

/**
 *
 * @author dov
 */
public class TestStuff {
    
    
    public static void main(String[] args) {
        try(
                Handle hand = new Handle(); 
                Matrix from = new Matrix(hand, new DArray(hand, 1,2,3,4,5,6,7,8,9,10,11,12), 3, 4);
                Matrix to = new Matrix(hand, 3, 4);
                ){
            
            System.out.println("input matrix\n" + from + "\n\noutput Matrix\n");
            
            KernelManager.get("neighborhoodSum").map(
                hand,
                from.dArray(), 4,
                to.dArray(), 1,
                3, 
                IArray.cpuTrue(), 
                IArray.cpuPointer(1)
            );
            
            System.out.println(to);
        }
    }
    
}
