package JCudaWrapper.array.Pointer.to2d;

import JCudaWrapper.array.Int.IArray;
import JCudaWrapper.array.Int.IArray2d;
import JCudaWrapper.array.Pointer.PArray;

/**
 *
 * @author E. Dov Neimand
 */
public interface PointerTo2d extends PArray{
 
    /**
     * An array that contains the pitch value of each pointed to array.
     * @return An array that contains the pitch value of each pointed to array.
     */
    public IArray targetLD();
    
    /**
     * The number of entries per line in the arrays pointed to.
     * @return The number of entries per line in the arrays pointed to.
     */
    public TargetDim2d targetDim();
    

    /**
     * Describes any line array, except that there's nto pointer information.
     */
    public class TargetDim2d{
        public final int entriesPerLine;
        public final int numLines;

        /**
         * Constructor
         * @param entriesPerLine The number of entries on each line.
         * @param numLines The number of lines.
         */
        public TargetDim2d(int entriesPerLine, int numLines) {
            this.entriesPerLine = entriesPerLine;
            this.numLines = numLines;
        }
        
        /**
         * The number of entries items.
         * @return The number of entries.
         */
        public int size(){
            return entriesPerLine*numLines;
        }
    }
}
