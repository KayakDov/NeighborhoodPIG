package JCudaWrapper.array.Int;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.LineArray;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public class IArray3d extends Array3d implements IArray{

    /**
     * 
     * @param entriesPerLine
     * @param linesPerLayer
     * @param numLayers
     * @param bytesPerEntry 
     */
    public IArray3d(int entriesPerLine, int linesPerLayer, int numLayers) {
        super(entriesPerLine, linesPerLayer, numLayers, Sizeof.INT);
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
    protected IArray3d(LineArray from, int startEntry, int entryPerLine, int startLayer, int numLayers) {
        super(from, startEntry, entryPerLine, startLayer, numLayers);
    }

    /**
     * Constructs this array from a 1d array.
     *
     * @param src The src array which shares memory with this array.
     * @param entriesPerLine The number of entries in each line.
     * @param linesPerLayer The number of lines in each layer.
     */
    public IArray3d(Array src, int entriesPerLine, int linesPerLayer) {
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
    public IArray3d(Array src, int entriesPerLine, int ld, int linesPerLayer) {
        super(src, entriesPerLine, ld, linesPerLayer);        
    }

    
    
    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array3d sub(int startEntry, int numEntryPerLine, int startLayer, int numLayers) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
    }

    /**
     * Guaranteed to throw an exception. TODO: implement this method.
     *
     * @throws UnsupportedOperationException always
     * @deprecated Unsupported operation.
     */
    @Override
    public Array1d depth(int entryIndex, int lineIndex) {
        throw new UnsupportedOperationException("Not supported yet."); // Generated from nbfs://nbhost/SystemFileSystem/Templates/Classes/Code/GeneratedMethodBody
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

}
