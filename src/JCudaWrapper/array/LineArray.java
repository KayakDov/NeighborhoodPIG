package JCudaWrapper.array;

import JCudaWrapper.array.Pointer.to2d.PSingletonTo2d;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.cudaPitchedPtr;

/**
 * Represents a 2D array where data is stored in contiguous memory sections,
 * with each section containing a fixed number of entries. This abstract class
 * provides foundational functionality for handling CUDA-pitched memory.
 *
 * <p>
 * Subclasses must ensure memory allocation for the {@code pointer} field.</p>
 *
 * @author E. Dov Neimand
 */
public abstract class LineArray implements Array {

    protected final cudaPitchedPtr pointer;
    public final int bytesPerEntry;

    
    
    /**
     * Constructs a LineArray with the specified memory layout. Be sure to
     * allocate the memory, as that is not done here.
     *
     * @param entriesPerLine The number of entries in each section of memory.
     * @param linesPerLayer The number of memory sections (lines) in a layer, where 2d arrays only have one layer.
     * @param bytesPerEntry The number of bytes per entry.
     */
    public LineArray(int entriesPerLine, int linesPerLayer, int bytesPerEntry) {//TODO: use entries per line
        this.pointer = new cudaPitchedPtr();
        pointer.ysize = linesPerLayer;
        pointer.xsize = entriesPerLine * bytesPerEntry;
        this.bytesPerEntry = bytesPerEntry;
        
    }
    
    /**
     * For restructuring an array from a pointer.
     * @param p The address of the array.
     * @param entriesPerLine The number of entries on each line of the array.
     * @param numLines The number of lines in the array.
     * @param bytesPerEntry The number of bytes for each entry in the array.
     * @param pitch The pitch of the array.
     */
    LineArray(Pointer p, int entriesPerLine, int numLines, int bytesPerEntry, int pitch){
        this(entriesPerLine, numLines, bytesPerEntry);
        pointer.ptr = p;
        pointer.pitch = pitch;
    }

    /**
     * Constructs a sub-array from an existing {@code LineArray}, starting from
     * a specified line and entry index.
     *
     * @param src The source {@code LineArray}.
     * @param startLine The starting line index.
     * @param numLines The number of lines to include.
     * @param startEntry The starting entry index within the line.
     * @param entriesPerLine The number of entries per line in the sub-array.
     */
    public LineArray(LineArray src, int startEntry, int entriesPerLine, int startLine, int numLines) {
        this(entriesPerLine, numLines, src.bytesPerEntry());

        pointer.ptr = src.get(startEntry, startLine).pointer();
        pointer.pitch = src.pitchedPointer().pitch;

    }

    /**
     * This array is creates as a sub array of the proffered array.
     *
     * @param src The super array.
     * @param entriesPerLine The number of entries on each included line. This
     * should divide into the number of elements.
     */
    public LineArray(Array src, int entriesPerLine) {
        this(src, entriesPerLine, entriesPerLine);
    }

    /**
     * This array is creates as a sub array of the proffered array.
     *
     * @param src The super array.
     * @param entriesPerLine The number of entries on each included line. This
     * should divide into the number of elements.
     * @param ld The number of entries that could be put in s line if the entire
     * pitch were used.
     */
    public LineArray(Array src, int entriesPerLine, int ld) {
        this(src, entriesPerLine, ld, src.size() / entriesPerLine);
    }
    
    
    /**
     * This array is creates as a sub array of the proffered array.
     *
     * @param src The super array.
     * @param entriesPerLine The number of entries on each included line. This
     * should divide into the number of elements.
     * @param ld The number of entries that could be put in s line if the entire
     * pitch were used.
     * @param linesPerLayer The number of lines on a layer.
     */
    public LineArray(Array src, int entriesPerLine, int ld, int linesPerLayer) {
        this(entriesPerLine, linesPerLayer, src.bytesPerEntry());
        pointer.ptr = src.pointer();
        pointer.pitch = ld * src.bytesPerEntry();
        
    }

    /**
     * The pitch of the pointer.
     */    
    public int pitch() {
        return (int) pointer.pitch;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int entriesPerLine() {
        return (int) pointer.xsize / bytesPerEntry();
    }

    /**
     * Retrieves an element at the specified section and entry index.
     *
     * @param lineNumber The index of the section.
     * @param indexInLine The index within the section.
     * @return A {@code Singleton} containing the desired element.
     */
    public Singleton get(int indexInLine, int lineNumber) {
        return get(lineNumber * entriesPerLine() + indexInLine);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int linesPerLayer() {
        return (int) pointer.ysize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int size() {
        return entriesPerLine() * linesPerLayer();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int bytesPerEntry() {
        return bytesPerEntry;
    }

    /**
     * {@inheritDoc}
     */
    public cudaPitchedPtr pitchedPointer() {
        return pointer;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Pointer pointer() {
        return pointer.ptr;
    }

    /**
     * {@inheritDoc}
     */
    public int ld() {
        return pitch() / bytesPerEntry();
    }

    
}
