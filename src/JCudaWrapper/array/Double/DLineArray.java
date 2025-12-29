package JCudaWrapper.array.Double;

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
        return new DArray1d(this, ld() * lineIndex, entriesPerLine(), 1);
    }

    /**
     * {@inheritDoc}
     */
    public default DSingleton getAt(int indexInLine, int lineNumber) {
        return new DSingleton(this, indexInLine + lineNumber * entriesPerLine());
    }

//    /**
//     * This method is depreciated since Kernel.run currently calls the kernel
//     * for PArray2dToD2d
//     *
//     */
//    @Override
//    public default DArray setProduct(Handle handle, double scalar, DArray src) {
//        Kernel.run("multiplyScalar", handle, size(), 
//                new PpArray2dTo2d[]{this},
//                P.to(src), P.to(src.ld()), P.to(src.entriesPerLine()), P.to(scalar));
//        return this;
//    }

}
