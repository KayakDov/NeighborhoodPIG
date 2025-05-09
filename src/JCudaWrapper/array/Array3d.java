package JCudaWrapper.array;

import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.runtime.cudaExtent;
import jcuda.runtime.cudaMemcpy3DParms;
import jcuda.runtime.cudaPitchedPtr;

/**
 * Abstract class representing a GPU array that stores data on the GPU. This
 * class provides various utilities for managing GPU memory and copying data
 * between the host (CPU) and device (GPU).
 *
 * <p>
 * It supports different primitive types including byte, char, double, float,
 * int, long, and short. The derived classes should implement specific
 * functionalities for each type of data.</p>
 *
 * <p>
 * Note: This class is designed to be extended by specific array
 * implementations.</p>
 *
 * The data is stored in a 3d row major format. The distance between the first
 * elements of each row, pointer.pitch >= width, is computed by JCuda.
 *
 * http://www.jcuda.org/tutorial/TutorialIndex.html#CreatingKernels
 *
 * TODO:Implement methods with JBLAS as an aulternative for when there's no gpu.
 *
 * @author E. Dov Neimand
 */
public abstract class Array3d extends LineArray implements Array {

    /**
     * The length of the array.
     */
    public final int layersPerGrid;

    /**
     * Constructs a 3d array.
     *
     * @param linesPerLayer The height of the memory to be allocated.
     * @param entriesPerLine The width of the memory to be allocated.
     * @param numLayers The depth of the memory to be allocated.
     * @param bytesPerEntry The memory size in bytes of elements stored in this
     * array.
     */
    protected Array3d(int entriesPerLine, int linesPerLayer, int numLayers, int bytesPerEntry) {

        super(entriesPerLine, linesPerLayer, bytesPerEntry);

        this.layersPerGrid = numLayers;
        opCheck(JCuda.cudaMalloc3D(
                pointer,
                new cudaExtent(entriesPerLine * bytesPerEntry, linesPerLayer, numLayers))
        );
        Array.recordMemAloc(pointer.ptr);
    }

    /**
     * Constructs a sub array. This sub array will have the same number of lines
     * per layer as this array, but may have other dimensions modified. This is
     * because a 3d array is essentially a line array.
     *
     * @param from The array this one is to be a sub array of.
     * @param startEntry The index of the first entry on each line.
     * @param entryPerLine The number of entries on each line.
     * @param startLayer The index of the first layer.
     * @param numLayers The number of layers.
     */
    protected Array3d(LineArray from, int startEntry, int entryPerLine, int startLayer, int numLayers) {
        super(from, 
                startEntry, entryPerLine, 
                startLayer * from.linesPerLayer(), from.linesPerLayer());

        this.layersPerGrid = numLayers;

        this.pointer.ptr = from.pointer(startEntry + startLayer * linesPerLayer() * ld());
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public Array3d(Array src, int entriesPerLine, int linesPerLayer) {
        this(src, entriesPerLine, entriesPerLine, linesPerLayer);
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param ld The number of elements that could be fit between the first
     * element of each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public Array3d(Array src, int entriesPerLine, int ld, int linesPerLayer) {
        super(src, entriesPerLine, ld, linesPerLayer);
        layersPerGrid = src.size() / (ld * linesPerLayer);
    }

    private cudaExtent extent;
    private cudaMemcpy3DParms getParams;
    private cudaMemcpy3DParms setParams;

    /**
     * Computes the extent, necessary for get and set parameters.
     *
     * @return The extent.
     */
    private cudaExtent extent() {
        if (extent == null)
            extent = new cudaExtent(
                    entriesPerLine() * bytesPerEntry(),
                    linesPerLayer(),
                    layersPerGrid
            );
        return extent;
    }

    /**
     * Sets memoryParms if it needs to be set, along with its fields srcPointer,
     * and extent.
     */
    private cudaMemcpy3DParms getParams() {
        if (getParams == null) {
            getParams = new cudaMemcpy3DParms();
            getParams.srcPtr = pitchedPointer();
            getParams.extent = extent();
            return getParams;
        } else
            return getParams;
    }

    /**
     * Sets memoryParms if it needs to be set, along with its fields dstPointer,
     * and extent.
     */
    private cudaMemcpy3DParms setParams() {
        if (setParams == null) {
            setParams = new cudaMemcpy3DParms();
            setParams.dstPtr = pitchedPointer();
            setParams.extent = extent();
            return setParams;
        } else
            return setParams;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Array dst) {
        getParams = getParams();

        getParams.kind = cudaMemcpyKind.cudaMemcpyDeviceToDevice;
        getParams.dstPtr
                = dst instanceof LineArray
                        ? ((LineArray) dst).pitchedPointer()
                        : ((Array1d) dst).pitchedPointer(entriesPerLine(), linesPerLayer());

        opCheck(JCuda.cudaMemcpy3DAsync(getParams, handle.getStream()));
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void get(Handle handle, Pointer dstCPUArray) {

        getParams = getParams();

        getParams.kind = cudaMemcpyKind.cudaMemcpyDeviceToHost;
        getParams.dstPtr = arrayPitchedPointer(dstCPUArray);

        opCheck(JCuda.cudaMemcpy3DAsync(getParams, handle.getStream()));
    }

    /**
     * Creates a pitched pointer for copying between this and a cpu array.
     *
     * @param cpuArray A pointer to the cpu array.
     * @return A pitched pointer for copying between this and a cpu array.
     */
    private cudaPitchedPtr arrayPitchedPointer(Pointer cpuArray) {
        cudaPitchedPtr pptr = new cudaPitchedPtr();
        pptr.ptr = cpuArray;
        pptr.pitch = (long) entriesPerLine() * bytesPerEntry();
        pptr.xsize = entriesPerLine() * bytesPerEntry();
        pptr.ysize = linesPerLayer();
        return pptr;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Array3d set(Handle handle, Pointer srcCPUArray) {

        setParams = setParams();

        setParams.kind = cudaMemcpyKind.cudaMemcpyHostToDevice;
        setParams.srcPtr = arrayPitchedPointer(srcCPUArray);

        opCheck(JCuda.cudaMemcpy3DAsync(setParams, handle.getStream()));
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return super.size() * layersPerGrid();
    }

    /**
     * Returns a pointer to the element at the specified 3D coordinates in the
     * pitched memory.
     *
     * @param entry The index of the entry in the desired line.
     * @param line The index of the desired line.
     * @param layer The index of the desired layer.
     * @return A Pointer pointing to the specified element in the pitched
     * memory.
     *
     * @throws ArrayIndexOutOfBoundsException if the coordinates are out of
     * bounds.
     */
    private Singleton get(int entry, int line, int layer) {
        return get(layer * linesPerLayer() * entriesPerLine() + line * entriesPerLine() + entry);
    }

    /**
     * The number of layers.
     *
     * @return The number of layers.
     */
    @Override
    public int layersPerGrid() {
        return layersPerGrid;
    }

    /**
     * Constructs a sub array.
     *
     * @param startEntry The index of the first entry on each line.
     * @param numEntryPerLine The number of entries on each line.
     * @param startLayer The index of the first layer.
     * @param numLayers The number of layers.
     * @return A sub array of this array.
     */
    abstract public Array3d sub(int startEntry, int numEntryPerLine, int startLayer, int numLayers);

    /**
     * The layer at the given index.
     *
     * @param index The index of the desired layer.
     * @return The layer at the given index.
     */
    public Array2d layer(int index) {
        return sub(0, entriesPerLine(), index, index + 1).as2d();
    }

    /**
     * The index on each layer at the given entry and line index.
     *
     * @param entryIndex The entry index in each line.
     * @param lineIndex The line index on each layer.
     * @return The index on each layer at the given entry and line index.
     */
    abstract public Array1d depth(int entryIndex, int lineIndex);

}
