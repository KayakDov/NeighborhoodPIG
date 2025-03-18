
package imageWork;

import JCudaWrapper.array.Float.FArray3d;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class ImageCreator extends Dimensions{

    /**
     * Array storing color data for each tensor element.
     */
    protected final String[] sliceNames;
    protected final String stackName;

    /**
     * 
     * @param cpuColors column major 
     * @param sliceNames
     * @param stackName
     * @param handle
     * @param image 
     */
    protected ImageCreator(String[] sliceNames, String stackName, Handle handle, FStrideArray3d image) {
        super(handle, image);
       
        this.sliceNames = sliceNames;
        this.stackName = stackName;
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
