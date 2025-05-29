
package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class HeatMapCreator {

    /**
     * Array storing color data for each tensor element.
     */
    protected final String[] sliceNames;
    protected final String stackName;
    protected final Dimensions dim;
    protected final Handle handle;

    /**
     * 
     * @param sliceNames The names of the slices.
     * @param stackName The name of the stack.
     * @param dim The dimensions of this heat map.
     */
    protected HeatMapCreator(String[] sliceNames, String stackName, Dimensions dim, Handle handle) {
        this.sliceNames = sliceNames;
        this.stackName = stackName;
        this.dim = dim;
        this.handle = handle;
    }

    /**
     * Displays the tensor data as a heat map in Fiji, supporting multiple
     * frames and depths.
     */
    public abstract void printToFiji();

    /**
     * Saves orientation heatmaps as images in the specified folder.
     *
     * @param writeToFolder The folder where images will be saved.
     */
    public abstract void printToFile(String writeToFolder);

    
}
