package fijiPlugin;


import ij.IJ;
import ij.plugin.PlugIn;
import jcuda.Pointer;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaError;
import org.apache.commons.math3.complex.Complex;

/**
 *
 * @author dov
 */
public class FijiPlugin implements PlugIn{


    @Override
    public void run(String string) {

        
        IJ.showMessage("Hello, World! ", " Welcome Ahhhh! Fiji plugin development! " );
    }
    
}
