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

//    public static void main(String[] args) {
//        try (
//                Handle hand = new Handle();
//                Matrix m = new Matrix(hand, 500, 500)) {
//
//            VectorsStride vs = m.rows();
//
//            for (int i = 0; i < m.getHeight(); i++) {
//                System.out.println(" i = " + i + " is " + vs.getVector(i).toString().substring(0, 100) + " ...");
//            }
//
////            System.out.println(m);
//        }
//    }
//    public static void main(String[] args) {
//        
//        int height =70, width = 90;
//        
//        System.out.println("Memory allocated = " + width * height);
//                
//        try(Handle hand = new Handle(); DArray array = DArray.empty(height*width)){            
//        
//            DStrideArray dsa = array.getAsBatch(1, height*(width - 1) + 1, height);
//            
//            for(int i = 0; i < height; i++){
//                System.out.println("\n\ni = " + i); 
//                
//                DArray da = dsa.getBatchArray(i);
//                
//                System.out.println(" has length " + da.length + " and is " + da.toString().substring(0, 100));
//            }
//        }
//    }
//    
//    public static void main(String[] args) {
//
//        int size = 1;
//
//        try (Handle hand = new Handle(); DArray array = DArray.empty(size)) {
//
//            for (int i = 0; i < 10_000_000; i++) {
//
//                System.out.println("\ni = " + i);
//
//                DArray da = array.subArray(0);
//                
//                System.out.println(da.toString());
//            }
//        }
//    }

//    public static void main(String[] args) {
//    
//        cudaStream_t stream = new cudaStream_t();
//        JCuda.cudaStreamCreate(stream);
//        
//        int totalSize = 1; 
//
//        Pointer devicePointer = new Pointer();
//        JCuda.cudaMalloc(devicePointer, totalSize * Sizeof.DOUBLE);
//
//        double[] subArrayHost = new double[1];
//
//        try {
//            for (int i = 0; i < 1_000_000; i++) {
//                System.out.println("\nIteration: " + i);
//                
//                Pointer subArrayPointer = devicePointer.withByteOffset(0);
//
//                JCuda.cudaMemcpyAsync(
//                        Pointer.to(subArrayHost),
//                        subArrayPointer,
//                        Sizeof.DOUBLE,
//                        cudaMemcpyKind.cudaMemcpyDeviceToHost,
//                        stream
//                );
//                
//                System.out.println(Arrays.toString(subArrayHost));
//            }
//        } finally {
//            // Free device memory and destroy the stream
//            JCuda.cudaFree(devicePointer);
//            JCuda.cudaStreamDestroy(stream);
//        }
//
//        System.out.println("Finished!");
//    }
    
    
    
    public static void main(String[] args) {
        try(
                Handle hand = new Handle(); 
                DArray d = new DArray(hand, 0, 1, 0, 2);
                DArray eigenVec = DArray.empty(2);
                DArray workSpace = DArray.empty(2);
                ){
            
            System.out.println(d);
            
            KernelManager.get("nullSpace1dBatch").map(
                hand,
                d, 2,
                eigenVec, 2,
                1, 
                IArray.cpuPointer(2), 
                DArray.cpuPointer(0.1)
            );
            
            System.out.println(eigenVec);
        }
    }
    
}
