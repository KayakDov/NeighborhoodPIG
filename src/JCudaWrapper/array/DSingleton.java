package JCudaWrapper.array;

import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 * Single element in the gpu.
 * @author E. Dov Neimand
 */
public class DSingleton extends Singleton implements DArray{

    /**
     * The first element in the proffered array is this singleton.
     * @param array An array from which the first element becomes this singleton.
     * @param index The index of the desired entry.
     */
    public DSingleton(DArray array, int index) {
        super(array, index);
    }

    /**
     * Creates an empty singleton. 
     */
    public DSingleton() {
        super(Sizeof.DOUBLE);
    }
    
    
    
    /**
     * Gets the element in this singleton.
     * @param handle
     * @return The element in this singleton.
     */
    public double getVal(Handle handle){
        double[] cpuArray = new double[1];
        get(handle, Pointer.to(cpuArray));
        handle.synch();
        return cpuArray[0];
    }
    
    /**
     * Sets the element in this singleton.
     * @param handle
     * @param d The new value in this singleton.
     * @return this.
     */
    public DSingleton set(Handle handle, double d){
        set(handle, P.to(d));
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public DSingleton set(Handle handle, DArray from) {
        super.set(handle, from);
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public DSingleton copy(Handle handle) {
        return new DSingleton().set(handle, this);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public DSingleton get(int i){
        return (DSingleton) super.get(i);
    }
    
}
