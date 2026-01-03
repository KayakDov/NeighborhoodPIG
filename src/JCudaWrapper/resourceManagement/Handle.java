package JCudaWrapper.resourceManagement;

import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.Pointer.to2d.PArray2dTo2d;
import JCudaWrapper.kernels.ThreadCount;
import fijiPlugin.Dimensions;
import jcuda.Pointer;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaStream_t;

/**
 * A utility class for managing a CUDA CUBLAS handle and a corresponding CUDA
 * stream. This class implements {@link AutoCloseable}, allowing for automatic
 * resource management when used in a try-with-resources statement. It handles
 * the creation, association, and destruction of a {@link cublasHandle} and
 * {@link cudaStream_t}, ensuring that resources are properly cleaned up.
 *
 *
 * Example usage:
 * <pre>
 * try (Handle handle = new Handle()) {
 *     // Use handle.get() to perform operations with the handle
 * }
 * </pre>
 *
 *
 * <p>
 * This class registers the handle and stream with a {@link Cleaner} to ensure
 * that both the CUBLAS handle and the CUDA stream are destroyed when the
 * {@link close()} method is called, either explicitly or implicitly.
 * </p>
 *
 * @author edov
 */
public class Handle implements AutoCloseable {

    /**
     * The CUBLAS handle for managing CUBLAS operations.
     */
    private cublasHandle handle;

    /**
     * The CUDA stream associated with the CUBLAS handle.
     */
    private cudaStream_t stream;

    private CUstream cuStream;
    
    private KernelManager km;

    /**
     * Constructs a new {@code Handle}, creating a CUBLAS handle and associating
     * it with a new CUDA stream. Both the handle and the stream are registered
     * for automatic cleanup.
     */
    public Handle() {
        // Initialize and create the CUBLAS handle
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);

        // Initialize and create the CUDA stream
        stream = new cudaStream_t();
        JCuda.cudaStreamCreate(stream);

        // Associate the CUBLAS handle with the CUDA stream
        JCublas2.cublasSetStream(handle, stream);
        
        km = new KernelManager();
    }

    /**
     * Returns the CUBLAS handle managed by this class.
     *
     * @return The {@link cublasHandle} for performing CUBLAS operations.
     */
    public cublasHandle get() {
        if(!isOpen) throw new RuntimeException("This handle has been closed.");
        if (cuStream != null) JCudaDriver.cuStreamSynchronize(cuStream);
        return handle;
    }

    /**
     * The stream associated with this handle.
     *
     * @return The stream associated with this handle.
     */
    public cudaStream_t getStream() {
        if(!isOpen) throw new RuntimeException("This handle has been closed.");
        if (cuStream != null) JCudaDriver.cuStreamSynchronize(cuStream);
        return stream;
    }

    /**
     * Synchronizes the stream managed by this group. The
     * {@code cudaStreamSynchronize} method is called on each stream to ensure
     * that all pending operations on not the stream are completed. This method
     * should be called before any methods not using the handle, like Array::set
     * or Array::get are called.
     *
     */
    public void synch() {
        JCuda.cudaStreamSynchronize(stream);
        if (cuStream != null) JCudaDriver.cuStreamSynchronize(cuStream);
    }


    /**
     * runs the kernel from the preset file in the KernelManagaer class
     * @param name The name of the function in the file.
     * @param numThreads The number of threads that will be run in the kernel
     * @param arrays All the arrays that will be passed to the kerenel.
     * @param dim The dimensions of the data.
     * @param additionalParmaters Pointers to any additional data.
     */
    public void runKernel(String name, ThreadCount numThreads, PArray2dTo2d[] arrays, Dimensions dim, Pointer... additionalParmaters){
        km.run(name, this, numThreads, arrays, dim, additionalParmaters);
    }
    
    /**
     * runs the kernel from the preset file in the KernelManagaer class
     * @param name The name of the function in the file.
     * @param arrays All the arrays that will be passed to the kerenel.
     * @param dim The dimensions of the data.
     * @param additionalParmaters Pointers to any additional data.
     */
    public void runKernel(String name, PArray2dTo2d[] arrays, Dimensions dim, Pointer... additionalParmaters){
        km.run(name, this, arrays, dim, additionalParmaters);
    }
    
    /**
     * A custream for this handle.
     *
     * @return
     */
    public CUstream cuStream() {
        if (cuStream == null) {
            cuStream = new CUstream();
            JCudaDriver.cuStreamCreate(cuStream, CUstream_flags.CU_STREAM_DEFAULT);            
        }
        
        JCuda.cudaStreamSynchronize(stream);
        
        return cuStream;
    }

    public boolean isOpen = true;
    
    /**
     * Closes the handle and stream, ensuring they are cleaned up. This method
     * is automatically called when the object is used in a try-with-resources
     * statement. It destroys the CUBLAS handle and the associated CUDA stream.
     */
    @Override
    public void close() {
        synch();
        JCublas2.cublasDestroy(handle);
        JCuda.cudaStreamDestroy(stream);
        isOpen = false;
        km.close();
        if (cuStream != null) JCudaDriver.cuStreamDestroy(cuStream);

    }
}
