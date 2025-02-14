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
}
