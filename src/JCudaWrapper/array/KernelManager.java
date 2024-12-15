package JCudaWrapper.array;

import JCudaWrapper.algebra.MatricesStride;
import JCudaWrapper.algebra.Vector;
import java.lang.ref.Cleaner;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import JCudaWrapper.resourceManagement.Handle;
import JCudaWrapper.resourceManagement.ResourceDealocator;
import java.io.File;

/**
 * The {@code Kernel} class is a utility for managing and executing CUDA kernels
 * using JCuda. It handles loading CUDA modules, setting up functions, and
 * executing them with specified parameters.
 *
 * Helper methods include
 * https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__DOUBLE.html
 * <p>
 * This class is designed to simplify the process of launching CUDA kernels with
 * a given input and output {@code DArray}.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>
 * {@code
 * Kernel kernel = new Kernel("atan2.ptx", "atan2xy", n);
 * DArray result = kernel.run(inputArray, outputArray, numOperations);
 * }
 * </pre>
 * </p>
 *
 * @author E. Dov Neimand
 */
public class KernelManager implements AutoCloseable {

    private static final GPUMath kernels = new GPUMath();
    /**
     * The CUDA function handle for the loaded kernel.
     */
    private final CUfunction function;

    /**
     * The number of threads per block to be used in kernel execution.
     */
    private final static int BLOCK_SIZE = 256;

    private CUmodule module;
    private final Cleaner.Cleanable cleanable;
    
    /**
     * The location of this directory.
     */
    private static Path path = Paths.get(KernelManager.class.getProtectionDomain().getCodeSource().getLocation().getPath()).getParent().getParent();



    /**
     * Constructs a {@code Kernel} object that loads a CUDA module from a given
     * file and retrieves a function handle for the specified kernel function.
     *
     * @param fileName The name of the PTX file containing the CUDA kernel
     * (e.g., "atan2.ptx"). If the file is in the kernel folder, then no path is
     * required.
     * @param functionName The name of the kernel function to execute (e.g.,
     * "atan2xy").
     * @param n The total number of operations to determine the grid size.
     */
    private KernelManager(String fileName, String functionName) {
        this.module = new CUmodule();

        File ptxFile = new File(path + File.separator + "src" + File.separator + "JCudaWrapper" + File.separator + "kernels" + File.separator + "ptx" + File.separator + fileName);
        if (!ptxFile.exists())
            throw new RuntimeException("Kernel file not found: " + ptxFile.getAbsolutePath());

        JCudaDriver.cuModuleLoad(module, ptxFile.getAbsolutePath());

        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);

        if (function == null) {
            throw new RuntimeException("Failed to load kernel function");
        }

        cleanable = ResourceDealocator.register(this, module -> JCudaDriver.cuModuleUnload(module), module);
    }

    /**
     * Gets a kernel and sets the batchsize to n. If this same kernel is being
     * used elsewhere with a different batchsize, it's changed there too.
     *
     * @param name not including the .ptx which the file must have. Also not
     * including any of the path so long as the file is in the kernel/ptx
     * folder.      *
     * @return The Kernel.
     */
    public static KernelManager get(String name) {
        return kernels.put(name);
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param <T> The type of array.
     * @param handle
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param incInput The increment for the input array.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @param incOutput The increment for the output array.
     * @param n The number of elements to be mapped.
     * @param additionalParmaters These should all be pointers to cpu arrays or
     * pointers to device pointers.
     * @return The {@code DArray} containing the processed results.
     */
    public <T extends Array> T map(Handle handle, int n, Array input, int incInput, T output, int incOutput, Pointer... additionalParmaters) {

        
        Pointer[] pointers = new Pointer[additionalParmaters.length + 3];
        
        pointers[0] = IArray.cpuPointer(incInput);
        pointers[1] = output.pToP();
        pointers[2] = IArray.cpuPointer(incOutput);
        
        
        if(additionalParmaters.length > 0) 
            System.arraycopy(additionalParmaters, 0, pointers, 3, additionalParmaters.length);
        
        map(handle, n, input, pointers);
        
        return output;
    }
    
    
    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param <T> The type of array.
     * @param handle
     * @param numThreads The number of threads to be used in the kernel.
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param additionalParmaters These should all be pointers to cpu arrays or
     * pointers to device pointers.
     */
    public <T extends Array> void map(Handle handle, int numThreads, Array input, Pointer... additionalParmaters) {

        
        NativePointerObject[] pointers = new NativePointerObject[additionalParmaters.length + 2];
        pointers[0] = IArray.cpuPointer(numThreads);
        pointers[1] = input.pToP();
        
        
        if(additionalParmaters.length > 0) 
            System.arraycopy(additionalParmaters, 0, pointers, 2, additionalParmaters.length);
        
        Pointer kernelParameters = Pointer.to(pointers);

        int gridSize = (int) Math.ceil((double) numThreads / BLOCK_SIZE);
        int result = JCudaDriver.cuLaunchKernel(
                function,
                gridSize, 1, 1, // Grid size (number of blocks)
                BLOCK_SIZE, 1, 1, // Block size (number of threads per block)
                0, handle.cuStream(), // Shared memory size and the specified stream
                kernelParameters, null // Kernel parameters
        );
        checkResult(result);

        JCudaDriver.cuCtxSynchronize();
    }
    
    /**
     * For operation between a vector and a batch of matrices.
     * @param hand
     * @param input
     * @param matrices
     * @return 
     */
    public DArray vectorBatchMatrix(Handle hand, Vector input, MatricesStride matrices){
        return map(hand, matrices.getBatchSize()*matrices.height*matrices.width, 
                input.dArray(), 
                input.inc(), 
                matrices.dArray(), 
                matrices.height, 
                IArray.cpuPointer(matrices.width),
                IArray.cpuPointer(matrices.colDist),
                IArray.cpuPointer(matrices.getStrideSize()));
    }
    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param <T> The type of array.
     * @param handle
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param incInput The increment for the input array.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @param incOutput The increment for the output array.
     * @param n The number of elements to be mapped.
     * @param shift additional parameters.
     * @return The {@code DArray} containing the processed results.
     */
    public <T extends Array> T map(Handle handle, T input, int incInput, T output, int incOutput, int n, int shift) {
        return map(handle, n, input, incInput, output, incOutput, IArray.cpuPointer(shift));
    }
    
    
    /**
     * Checks for error messages, and throws an exception if the operation
     * failed.
     *
     * @param result The result of a cuLaunch.
     */
    private void checkResult(int result) {
        if (result != CUresult.CUDA_SUCCESS) {
            String[] errorMsg = new String[1];
            JCudaDriver.cuGetErrorString(result, errorMsg);
            throw new RuntimeException("CUDA error during : " + errorMsg[0]);
        }
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param handle
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @param n The number of elements to be mapped.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(Handle handle, DArray input, DArray output, int n) {
        return map(handle, n, input, 1, output, 1);
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream. Note, a stream is generated for this method, so be
     * sure that the data is synchronized before and after.
     *
     * @param handle
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @param output The {@code DArray} representing the output data where
     * results will be stored.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray map(Handle handle, Vector input, Vector output) {
        return map(handle, Math.min(input.dim(), output.dim()), input.dArray(), input.inc(), output.dArray(), output.inc());
    }

    /**
     * Runs the loaded CUDA kernel with the specified input and output arrays on
     * a specified stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapping(Handle handle, DArray input) {
        return map(handle, input, DArray.empty(input.length), input.length);
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param handle The handle.
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(Handle handle, DArray input) {
        return map(handle, input, input, input.length);
    }
    

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(Handle handle, DArray input, int inc, int n) {
        return map(handle, n, input, inc, input, inc);
    }

    /**
     * Runs the loaded CUDA kernel, mapping the input to itself, on a specified
     * stream.
     *
     * @param handle The handle.
     * @param input The {@code DArray} representing the input data to be
     * processed by the kernel.
     * @return The {@code DArray} containing the processed results.
     */
    public DArray mapToSelf(Handle handle, Vector input) {
        return map(handle, input, input);
    }

    /**
     * Cleans up resources by unloading the CUDA module.
     */
    public void close() {
        cleanable.clean();
    }

    /**
     * Manages the kernels that implement commonly used math functions.
     *
     * @author E. Dov Neimand
     */
    private static class GPUMath extends HashMap<String, KernelManager> implements AutoCloseable {

        /**
         * Puts a kernel with the given properties in the map.
         *
         * @param name The name of the kernel's file. Don't include the ptx.
         * file. The file should be in the folder src/kernels/ptx and no path is
         * required.
         * @param functionName The name of the function in the kernel to be
         * called.
         * @return The kernel with the given properties.
         */
        public KernelManager put(String name) {

            String fileName = name + ".ptx", functionName = name + "Kernel";

            KernelManager put = get(name);
            if (put == null) {
                put = new KernelManager(fileName, functionName);
                put(name, put);
            }
            return put;
        }

        @Override
        public void close() {
            values().forEach(k -> k.close());
        }

    }
    
    
    public static void main(String[] args) {
        try(Handle hand = new Handle(); DArray array = new DArray(hand, 4, 9, 16, 25); DArray target = DArray.empty(4)){
            KernelManager.get("sqrt").mapToSelf(hand, array);
            System.out.println(array);
        }
    }

}
