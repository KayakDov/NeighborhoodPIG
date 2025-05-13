package JCudaWrapper.array;

import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray1d;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import java.awt.Dimension;
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
public class Kernel implements AutoCloseable {

    /**
     * The CUDA function handle for the loaded kernel.
     */
    private final CUfunction function;

    /**
     * The number of threads per block to be used in kernel execution.
     */
    private final static int BLOCK_SIZE = 256;

    private CUmodule module;//TODO: set this up so that multiple kernels can use the same module.

    public enum Type {

        DOUBLE("double"), FLOAT("float"), PTR_TO_DOUBLE_2D("P2dToD2d");

        public final String folder;

        /**
         * Constructs the DataType
         *
         * @param parentFileName The name of the parent file that has the
         * classes dealing with this datatype.
         */
        private Type(String parentFileName) {
            this.folder = parentFileName;
        }
    }

    /**
     * Constructs a {@code Kernel} object that loads a CUDA module from a given
     * file and retrieves a function handle for the specified kernel function.
     *
     * @param name The name of the file without the .cu or .ptx at the end of
     * it. This should also be the name of the main function in the kernel with
     * the work "Kernel" appended.
     * @param dataType The type of data the file operates on.
     */
    public Kernel(String name, Type dataType) {
        String fileName = name + ".ptx", functionName = name + "Kernel";
        this.module = new CUmodule();

        try (InputStream resourceStream = getClass().getClassLoader()
                .getResourceAsStream("JCudaWrapper/kernels/ptx/" + dataType.folder + "/" + fileName)) {

            if (resourceStream == null)
                throw new RuntimeException("Kernel file not found in JAR: " + fileName);

            File tempFile = File.createTempFile("kernel_", ".ptx");
            tempFile.deleteOnExit(); // Clean up after the program ends
            Files.copy(resourceStream, tempFile.toPath(), StandardCopyOption.REPLACE_EXISTING);  //TODO: copying the file seems ineficiant.  Can this be made faster?

            JCudaDriver.cuModuleLoad(module, tempFile.getAbsolutePath());

        } catch (Exception e) {
            throw new RuntimeException("Failed to load kernel file", e);
        }

        function = new CUfunction();
        JCudaDriver.cuModuleGetFunction(function, module, functionName);

        if (function == null) {
            throw new RuntimeException("Failed to load kernel function");
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
     * @param source The array the kernel will run on.
     * @param additionalArguments An array of pointers to all additional
     * arguments.
     */
    public static void run(String name, Handle handle, int numThreads, Array source, Pointer... additionalArguments) {//TODO: organize this so vectors are always followed by their ld and height.

        try (Kernel km = new Kernel(name, Type.PTR_TO_DOUBLE_2D)) {
            km.run(handle, numThreads, source, additionalArguments);
        }
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
    public <T extends Array> void run(Handle handle, int numThreads, T input, Pointer... additionalParmaters) {

        NativePointerObject[] pointers = new NativePointerObject[additionalParmaters.length + 2];
        pointers[0] = P.to(numThreads);
        pointers[1] = P.to(input);

        if (additionalParmaters.length > 0)
            System.arraycopy(additionalParmaters, 0, pointers, 2, additionalParmaters.length);

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
    public static void run(String name, Handle handle, int numThreads, PArray2dTo2d[] arrays, Dimensions dim, Pointer... additionalParmaters) {

        try (Kernel km = new Kernel(name, Type.PTR_TO_DOUBLE_2D)) {
            km.run(handle, numThreads, arrays, dim, additionalParmaters);
        }
    }

    /**
     * A cleaner format for calling kernels using lots of PArray2dTo2d.
     *
     * @param handle The context.
     * @param numThreads The total number of threads.
     * @param arrays Each of the arrays being used. Be sure their order matches
     * their use in the kernel.
     * @param dim The dimensions as generated by the dimension class.
     * @param additionalParmaters Any additional parameters needed.
     */
    public void run(Handle handle, int numThreads, PArray2dTo2d[] arrays, Dimensions dim, Pointer... additionalParmaters) {

        try (IArray dimArray = dim.gpuDim()) {

            Pointer[] pointers = new Pointer[additionalParmaters.length + 4 * arrays.length - 1 + 3];

            int pointInd = 0;
            pointers[pointInd++] = P.to(arrays[0].targetLD());
            pointers[pointInd++] = P.to(arrays[0].targetLD().ld());
            pointers[pointInd++] = P.to(arrays[0].ld());

            for (int i = 1; i < arrays.length; i++) {
                pointers[pointInd++] = P.to(arrays[i]);
                pointers[pointInd++] = P.to(arrays[i].targetLD());
                pointers[pointInd++] = P.to(arrays[i].targetLD().ld());
                pointers[pointInd++] = P.to(arrays[i].ld());
            }
            pointers[pointInd++] = P.to(dimArray);

            System.arraycopy(additionalParmaters, 0, pointers, pointInd, pointers.length);

            run(handle, numThreads, arrays[0], additionalParmaters);
        }
    }

    /**
     * Checks for error messages, and throws an exception if the operation
     * failed.
     *
     * @param result The result of a cuLaunch.
     */
    private void checkResult(int result) {
        int err = JCuda.cudaGetLastError();
        if (err != cudaError.cudaSuccess)
            throw new RuntimeException("CUDA error during : " + JCuda.cudaGetErrorString(err));

        if (result != CUresult.CUDA_SUCCESS) {
            String[] errorMsg = new String[1];
            JCudaDriver.cuGetErrorString(result, errorMsg);
            throw new RuntimeException("CUDA error during : " + errorMsg[0]);
        }
    }

    /**
     * Cleans up resources by unloading the CUDA module.
     */
    @Override
    public void close() {
        JCudaDriver.cuModuleUnload(module);
    }

}
