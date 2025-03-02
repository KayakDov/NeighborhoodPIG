package JCudaWrapper.array.Float;

import JCudaWrapper.array.P;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class FSingleton extends Singleton implements FArray{


    /**
     * The first element in the proffered array is this singleton.
     * @param array An array from which the first element becomes this singleton.
     * @param index The index of the desired entry.
     */
    public FSingleton(FArray array, int index) {
        super(array, index);
    }

    /**
     * Creates an empty singleton. 
     */
    public FSingleton() {
        super(Sizeof.FLOAT);
    }
    
    
    
    /**
     * Gets the element in this singleton.
     * @param handle
     * @return The element in this singleton.
     */
    public float getVal(Handle handle){
        float[] cpuArray = new float[1];
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
    public FSingleton set(Handle handle, float d){
        set(handle, P.to(d));
        return this;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FSingleton set(Handle handle, FArray from) {
        super.set(handle, from);
        return this;
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public FSingleton copy(Handle handle) {
        return new FSingleton().set(handle, this);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public FSingleton get(int i){
        return (FSingleton) super.get(i);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public float[] get(Handle handle) {
        float[] cpuArray = new float[size()];
        get(handle, Pointer.to(cpuArray));
        return cpuArray;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public FSingleton setProduct(Handle handle, float scalar, FArray src) {
        as1d().setProduct(handle, scalar, src);
        return this;
    }    
}
