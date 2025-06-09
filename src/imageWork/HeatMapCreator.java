
package imageWork;

import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.ImageStack;

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
    public void printToFiji(){
        getIP().show();
    }

    /**
     * Saves orientation heatmaps as images in the specified folder.
     *
     * @param writeToFolder The folder where images will be saved.
     */
    public abstract void printToFile(String writeToFolder);

    /**
     * An imagePlus of the image.
     * @return An imagePlus of the image.
     */
    public ImagePlus getIP(){
        return dim.setToHyperStack(new ImagePlus(stackName, getStack()));
    }
    
    /**
     * Gets the image stack;
     * @return The image stack;
     */
    public abstract ImageStack getStack();
    
}
