package JCudaWrapper.array;

import JCudaWrapper.array.Pointer.to2d.PSingletonTo2d;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import JCudaWrapper.resourceManagement.Handle;
import java.util.HashSet;
import jcuda.driver.JCudaDriver;

/**
 * Represents a 2D GPU array with memory allocated on the device. This class
 * provides methods to interact with the allocated memory and copy data between
 * device and host.
 *
 * @author E. Dov Neimand
 */
public abstract class Array2d extends LineArray {

    /**
     * Constructs a 2D array in GPU memory.
     *
     * @param numLines The number of sections to be allocated.
     * @param entriesPerLine The number of entries in each section.
     * @param bytesPerEntry The memory size in bytes of elements stored in this
     * array.
     */
    public Array2d(int entriesPerLine, int numLines, int bytesPerEntry) {
        super(entriesPerLine, numLines, bytesPerEntry);

        long[] pitchArray = new long[1];

        opCheck(JCuda.cudaMallocPitch(pointer.ptr, pitchArray, entriesPerLine * bytesPerEntry, linesPerLayer()));
        Array.recordMemAloc(pointer.ptr);
        
        pointer.pitch = pitchArray[0];
        
        
    }

    /**
     * This array is creates as a sub array of the proffered array.
     *
     * @param src The super array.
     * @param startLine The line in src that this array starts.
     * @param numLines The number of lines in this array.
     * @param startEntry The start index on each included line.
     * @param entriesPerLine The number of entries on each included line.
     */
    public Array2d(LineArray src, int startEntry, int entriesPerLine, int startLine, int numLines) {
        super(src, startEntry, entriesPerLine, startLine, numLines);
    }

    /**
     * Constructs this 2d array from a1d array. Note, the data is not copied,
     * and changes to one array will effect the other.
     *
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line.
     */
    public Array2d(Array src, int entriesPerLine) {
        super(src, entriesPerLine);
    }
    
    /**
     * Creates a 2d array from a pointer to a 2d array.
     * @param hand
     * @param to2d The singleton with a pointer to the array.
     */
    public Array2d(Pointer to2d, int entriesPerLine, int numLines, int ld, int bytesPerEntry){
        super(to2d, entriesPerLine, numLines, bytesPerEntry, ld);
    }

    /**
     * Constructs this 2d array from a1d array. Note, the data is not copied,
     * and changes to one array will effect the other.
     *
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line.
     * @param ld The number of entries that could be put on a line if the entire
     * pitch were used.
     */
    public Array2d(Array src, int entriesPerLine, int ld) {
        super(src, entriesPerLine, ld);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Pointer dstCPUArray) {
        int width = entriesPerLine() * bytesPerEntry();
        opCheck(JCuda.cudaMemcpy2DAsync(
                dstCPUArray,
                width,
                pointer(),
                pitch(),
                width,
                linesPerLayer(),
                cudaMemcpyKind.cudaMemcpyDeviceToHost,
                handle.getStream()
        ));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d set(Handle handle, Pointer srcCPUArray) {
        
        int width = entriesPerLine() * bytesPerEntry();
                
        opCheck(JCuda.cudaMemcpy2DAsync(
                pointer(),
                pitch(),
                srcCPUArray,
                width,
                width,
                linesPerLayer(),
                cudaMemcpyKind.cudaMemcpyHostToDevice,
                handle.getStream()
        ));
        
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Array dst) {
        opCheck(JCuda.cudaMemcpy2DAsync(
                dst.pointer(),
                (dst instanceof LineArray? ((LineArray)dst).pitch() : entriesPerLine()*bytesPerEntry),
                pointer(),
                pitch(),
                entriesPerLine() * bytesPerEntry(),
                linesPerLayer(),
                cudaMemcpyKind.cudaMemcpyDeviceToDevice,
                handle.getStream()
        ));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array2d set(Handle handle, Array from) {
        return (Array2d) super.set(handle, from); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/OverriddenMethodBody
    }

    /**
     * The line with the requested index.
     * @param index The index of the desired line.
     * @return The line with the requested index.
     */
    public Array1d line(int index){
        return as1d().sub(index*ld(), entriesPerLine());
    }
    
    /**
     * All the entries at the requested index in their lines.
     * @param index The index of the desired entries.
     * @return All the entries at the requested index in their lines.
     */
    public Array1d entriesAt(int index){
        
        return as1d().sub(index, ld()*(linesPerLayer() - 1) + 1, ld());
    }
}
