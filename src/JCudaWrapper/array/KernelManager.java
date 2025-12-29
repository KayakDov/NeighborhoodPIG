package JCudaWrapper.array;

import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import java.io.Closeable;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

/**
 * The {@code Kernel} class is a utility for managing and executing CUDA kernels
 * using JCuda. It handles loading CUDA modules, setting up functions, and
 * executing them with specified parameters.
 *
 * TODO: change kernel to take advantage of matrices structure. Rewrite them for
 * 2d indexing.
 * <p>
 * Example usage:
 * <pre>
 * {@code
 * Kernel kernel = new Kernel("atan2");
 * DArray result = kernel.run(handle, numOperations, inputArray, P.to(outputArray));
 * }
 * </pre>
 * </p>
 *
 * @author E. Dov Neimand
 */
public class KernelManager implements AutoCloseable {

    private CUmodule module;
    private final static int BLOCK_SIZE = 256;
    private static final String CONSOLIDATED_PTX = "JCudaWrapper/kernels/ptx/kernels.ptx";

    public KernelManager() {

        module = new CUmodule();
        try (InputStream resourceStream = JCudaWrapper.array.KernelManager.class.getClassLoader()
                .getResourceAsStream(CONSOLIDATED_PTX)) {

            if (resourceStream == null) {
                throw new RuntimeException("Resource not found: " + CONSOLIDATED_PTX);
            }

            File tempFile = File.createTempFile("kernel_", ".ptx");
            tempFile.deleteOnExit();
            Files.copy(resourceStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);

            // Note: We use a separate result check since we're in a static context
            int result = JCudaDriver.cuModuleLoad(module, tempFile.getAbsolutePath());
            if (result != CUresult.CUDA_SUCCESS) {
                throw new RuntimeException("CUDA Driver error " + result + " loading module.");
            }

        } catch (Exception e) {
            throw new RuntimeException("Failed to load consolidated kernels: " + e.getMessage(), e);
        }
    }

    /**
     * Runs the kernel on the parameters provided. If the kernel will only be
     * run once, use this method. If the kernel will be used repeatedly, use the
     * constructor and remember to close it.
     *
     * @param name The name of the kernel file without the .cu. The main
     * function in the kernel must me named nameKernel( ...
     * @param handle The handle to be used to make the kernel run.
     * @param numThreads The number of threads the kernel will run.
     * @param additionalArguments An array of pointers to all additional
     * arguments.
     */
    public void run(String name, Handle handle, int numThreads, Pointer... additionalArguments) {//TODO: organize this so vectors are always followed by their ld and height.

        new Kernel(name).run(handle, numThreads, additionalArguments);

    }

    /**
     * Runs the kernel on the parameters provided.If the kernel will only be run
     * once, use this method.If the kernel will be used repeatedly, use the
     * constructor and remember to close it.
     *
     * @param name The name of the kernel file without the .cu. The main
     * function in the kernel must me named nameKernel( ...
     * @param handle The handle to be used to make the kernel run.
     * @param numThreads The number of threads the kernel will run.
     * @param arrays The arrays to be passed.
     * @param dim The dimensions of the arrays.
     * @param additionalParmaters Any additional parameters.
     */
    public void run(String name, Handle handle, int numThreads, PArray2dTo2d[] arrays, Dimensions dim, Pointer... additionalParmaters) {

        new Kernel(name).run(handle, numThreads, arrays, dim.getGpuDim(), additionalParmaters);
    }

    public class Kernel {

        /**
         * The CUDA function handle for the loaded kernel.
         */
        private final CUfunction function;

        /**
         * Sets up a kernel.
         *
         * @param functionName The name of the file, without the .cu.
         */
        public Kernel(String functionName) {

            function = new CUfunction();
            checkResult(JCudaDriver.cuModuleGetFunction(function, module, functionName));
        }

        /**
         * Runs the loaded CUDA kernel with the specified input and output
         * arrays on a specified stream.Note, a stream is generated for this
         * method, so be sure that the data is synchronized before and after.
         *
         * @param <T> The type of array.
         * @param handle
         * @param numThreads The number of threads to be used in the kernel.
         * @param additionalParmaters These should all be pointers to cpu arrays
         * or pointers to device pointers.
         */
        public <T extends Array> void run(Handle handle, int numThreads, Pointer... additionalParmaters) {

            NativePointerObject[] pointers = new NativePointerObject[additionalParmaters.length + 1];
            pointers[0] = P.to(numThreads);

            System.arraycopy(additionalParmaters, 0, pointers, 1, additionalParmaters.length);

            Pointer kernelParameters = Pointer.to(pointers);

            int gridSize = (int) Math.ceil((double) numThreads / BLOCK_SIZE);

            checkResult(JCudaDriver.cuLaunchKernel(
                    function,
                    gridSize, 1, 1, // Grid size (number of blocks)
                    BLOCK_SIZE, 1, 1, // Block size (number of threads per block)
                    0, handle.cuStream(), // Shared memory size and the specified stream
                    kernelParameters, null // Kernel parameters
            ));

            JCudaDriver.cuCtxSynchronize();
        }

        /**
         * A cleaner format for calling kernels using lots of PArray2dTo2d.
         *
         * @param handle The context.
         * @param numThreads The total number of threads.
         * @param arrays Each of the arrays being used. Be sure their order
         * matches their use in the kernel. If this array is empty then an
         * exception will be thrown.
         * @param dim The dimensions as generated by the dimension class.
         * @param additionalParmaters Any additional parameters needed.
         */
        public void run(Handle handle, int numThreads, PArray2dTo2d[] arrays, IArray dim, Pointer... additionalParmaters) {

            Pointer[] pointers = new Pointer[additionalParmaters.length + 4 * arrays.length + 1];

            int pointInd = 0;

            for (int i = 0; i < arrays.length; i++) {
                pointers[pointInd++] = P.to(arrays[i]);
                pointers[pointInd++] = P.to(arrays[i].targetLD());
                pointers[pointInd++] = P.to(arrays[i].targetLD().ld());
                pointers[pointInd++] = P.to(arrays[i].ld());
            }
            pointers[pointInd++] = P.to(dim);

            System.arraycopy(additionalParmaters, 0, pointers, pointInd, additionalParmaters.length);

            run(handle, numThreads, pointers);

        }

        /**
         * Checks for error messages, and throws an exception if the operation
         * failed.
         *
         * @param result The result of a cuLaunch.
         */
        private void checkResult(int result) {
            int err = JCuda.cudaGetLastError();
            if (err != cudaError.cudaSuccess) {
                throw new RuntimeException("CUDA error during : " + JCuda.cudaGetErrorString(err));
            }

            if (result != CUresult.CUDA_SUCCESS) {
                String[] errorMsg = new String[1];
                JCudaDriver.cuGetErrorString(result, errorMsg);
                throw new RuntimeException("CUDA error during : " + errorMsg[0]);
            }
        }

    }

    public void close() {
        if (module != null) {
            JCudaDriver.cuModuleUnload(module);
        }
        module = null;
    }
}
