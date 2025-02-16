package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public interface DLineArray extends DArray {

    /**
     *
     * @param lineIndex
     * @return
     */
    public default DArray1d getLine(int lineIndex) {
        return new DArray1d(this, ld() * lineIndex, entriesPerLine());
    }
    
    /**
     * {@inheritDoc}
     */
    public default DSingleton getAt(int indexInLine, int lineNumber) {
        return new DSingleton(this, indexInLine + lineNumber*entriesPerLine());
    }
}
