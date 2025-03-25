
package imageWork;

import JCudaWrapper.array.Float.FArray3d;
import JCudaWrapper.array.Float.FStrideArray3d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import ij.ImagePlus;
import ij.plugin.HyperStackConverter;

/**
 *
 * @author E. Dov Neimand
 */
public abstract class HeatMapCreator extends Dimensions{

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
    protected HeatMapCreator(String[] sliceNames, String stackName, Handle handle, FStrideArray3d image) {
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
    
    public ImagePlus setToHyperStack(ImagePlus imp){
        if (batchSize > 1) {
            imp = HyperStackConverter.toHyperStack(
                    imp,
                    1,
                    depth,
                    batchSize
            );
        }
        return imp;
    }
        
}
