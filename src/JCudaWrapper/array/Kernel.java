package JCudaWrapper.array;

import jcuda.NativePointerObject;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;
import jcuda.driver.JCudaDriver;
import JCudaWrapper.resourceManagement.Handle;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;

/**
 * The {@code Kernel} class is a utility for managing and executing CUDA kernels
 * using JCuda. It handles loading CUDA modules, setting up functions, and
 * executing them with specified parameters.
 *
 *
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

    /**
     * Constructs a {@code Kernel} object that loads a CUDA module from a given
     * file and retrieves a function handle for the specified kernel function.
     *
     * @param name The name of the file without the .cu or .ptx at the end of
     * it. This should also be the name of the main function in the kernel with
     * the work "Kernel" appended.
     */
    public Kernel(String name) {
        String fileName = name + ".ptx", functionName = name + "Kernel";
        this.module = new CUmodule();

        try (InputStream resourceStream = getClass().getClassLoader()
                .getResourceAsStream("JCudaWrapper/kernels/ptx/" + fileName)) {
            
            if (resourceStream == null) throw new RuntimeException("Kernel file not found in JAR: " + fileName);
            
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
    public static void run(String name, Handle handle, int numThreads, Array source, Pointer... additionalArguments) {

        try (Kernel km = new Kernel(name)) {
            km.map(handle, numThreads, source, additionalArguments);
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
    public <T extends Array> void map(Handle handle, int numThreads, T input, Pointer... additionalParmaters) {

        NativePointerObject[] pointers = new NativePointerObject[additionalParmaters.length + 2];
        pointers[0] = P.to(numThreads);
        pointers[1] = P.to(input);

        if (additionalParmaters.length > 0)
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
     * Cleans up resources by unloading the CUDA module.
     */
    @Override
    public void close() {
        JCudaDriver.cuModuleUnload(module);
    }

}
