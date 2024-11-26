
package main;

import JCudaWrapper.algebra.Matrix;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

/**
 *
 * @author dov
 */
public class TestStuff {

    public static void main(String[] args) {
        try(
                Handle hand = new Handle(); 
                Matrix m = new Matrix(hand, 500, 500)){
            System.out.println(m);
        }

    }
}
