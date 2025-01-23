package fijiPlugin;

import JCudaWrapper.array.DArray;
import java.io.File;
import java.util.Iterator;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;

/**
 * This class takes in an image source and breaks it down into one or more
 * DArrays for processing.
 *
 * @author E. Dov Neimand
 */
public class ImgToGPU implements Iterator<DArray> {

    private final int depth, height, width;    
    private final int framesPerArray;
    private final int numFrames;

    private Iterator<double[]> img;

    public ImgToGPU(int depth, int height, int width, Iterator<double[]> img, int numImages) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.img = img;
        long bytesPerArray = getGpuMemoryInfo()[0]/20;
        long bytesPerImage = Double.SIZE * width * height;
        long bytesPerFrame = bytesPerImage*depth;
        framesPerArray = (int)(bytesPerArray/bytesPerFrame);
        numFrames = numImages/depth;
    }

    
    
    /**
     * Returns the amount of free and total memory available on the GPU.
     *
     * @return An array where index 0 is the free memory (bytes) and index 1 is
     * the total memory (bytes).
     * @throws RuntimeException If there is an error retrieving GPU memory
     * information.
     */
    public static long[] getGpuMemoryInfo() {
        long[] freeMemory = new long[1];
        long[] totalMemory = new long[1];

        int result = JCuda.cudaMemGetInfo(freeMemory, totalMemory);
        if (result != cudaError.cudaSuccess) 
            throw new RuntimeException("Failed to get GPU memory info: " + JCuda.cudaGetErrorString(result));        

        return new long[]{freeMemory[0], totalMemory[0]};
    }

    @Override
    public boolean hasNext() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    @Override
    public DArray next() {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

}
