package JCudaWrapper.array.Int;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.LineArray;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class IArray2d extends Array2d implements IArray{

    /**
     * Constructs a 2 dimensional Iarray.
     * @param entriesPerLine THe number of entries in each line.
     * @param numLines The number of lines in the array.
     */
    public IArray2d(int entriesPerLine, int numLines) {
        super(entriesPerLine, numLines, Sizeof.INT);
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
    public IArray2d(LineArray src, int startEntry, int entriesPerLine, int startLine, int numLines) {
        super(src, startEntry, entriesPerLine, startLine, numLines);
    }

    /**
     * Constructs this 2d array from a1d array. Note, the data is not copied,
     * and changes to one array will effect the other.
     *
     * @param src The 1d array.
     * @param entriesPerLine The number of entries per line.
     */
    public IArray2d(Array src, int entriesPerLine) {
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
    public IArray2d(Array src, int entriesPerLine, int ld) {
        super(src, entriesPerLine, ld);
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array copy(Handle handle) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    
    /**
     * Takes a preallocated pointer and gives it an array structure.
     * @param to2d The target of the singleton's pointer.
     * @param entriesPerLine The number of entries on each line.
     * @param numLines The number of lines.
     * @param ld The leading dimension of each entry.
     */
    public IArray2d(Pointer to2d, int entriesPerLine, int numLines, int ld) {
        super(to2d, entriesPerLine, numLines, ld, Sizeof.INT);
    }
    
}
