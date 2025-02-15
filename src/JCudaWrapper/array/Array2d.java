package JCudaWrapper.array;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import JCudaWrapper.resourceManagement.Handle;

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
    public Array2d(int numLines, int entriesPerLine, int bytesPerEntry) {
        super(entriesPerLine, numLines, bytesPerEntry);

        long[] pitchArray = new long[1];

        opCheck(JCuda.cudaMallocPitch(pointer.ptr, pitchArray, entriesPerLine() * bytesPerEntry, linesPerLayer()));        
        
        pointer.pitch = pitchArray[0];
        pointer.xsize = entriesPerLine();
        pointer.ysize = linesPerLayer();        
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
    public Array2d(LineArray src, int startLine, int numLines, int startEntry, int entriesPerLine) {
        super(src, startLine, numLines, startEntry, entriesPerLine);
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
        opCheck(JCuda.cudaMemcpy2DAsync(
                dstCPUArray,
                bytesPerLine(),
                pointer(),
                bytesPerLine(),
                entriesPerLine() * bytesPerEntry(),
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
        opCheck(JCuda.cudaMemcpy2DAsync(
                pointer(),
                bytesPerLine(),
                srcCPUArray,
                bytesPerLine(),
                entriesPerLine() * bytesPerEntry(),
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
        opCheck(JCuda.cudaMemcpy2DAsync(dst.pointer(),
                dst.bytesPerLine(),
                pointer(),
                bytesPerLine(),
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
        return as1d().sub(index, ld()*(linesPerLayer() - 1), ld());
    }
}
