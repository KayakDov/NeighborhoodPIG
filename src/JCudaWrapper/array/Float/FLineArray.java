package JCudaWrapper.array.Float;

import JCudaWrapper.array.Double.*;
import JCudaWrapper.array.Kernel;
import JCudaWrapper.array.P;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Sizeof;

/**
 *
 * @author E. Dov Neimand
 */
public interface FLineArray extends FArray {

    /**
     *
     * @param lineIndex
     * @return
     */
    public default FArray1d getLine(int lineIndex) {
        return new FArray1d(this, ld() * lineIndex, entriesPerLine());
    }
    
    /**
     * {@inheritDoc}
     */
    public default FSingleton getAt(int indexInLine, int lineNumber) {
        return new FSingleton(this, indexInLine + lineNumber*entriesPerLine());
    }

//    /**
//     * {@inheritDoc }
//     */
//    @Override
//    public default FArray setProduct(Handle handle, float scalar, FArray src){
//        Kernel.run("multiplyScalar", handle, size(), this, P.to(ld()), P.to(entriesPerLine()), 
//                P.to(src), P.to(src.ld()), P.to(src.entriesPerLine()), P.to(scalar));
//        return this;
//    }
    
    
    
}
