/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package JCudaWrapper.array.Int;

import JCudaWrapper.array.Array;
import JCudaWrapper.array.Array1d;
import JCudaWrapper.array.Array2d;
import JCudaWrapper.array.Array3d;
import JCudaWrapper.array.Double.DArray;
import JCudaWrapper.array.Double.DSingleton;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Singleton;
import JCudaWrapper.resourceManagement.Handle;
import jcuda.Pointer;
import jcuda.Sizeof;

/**
 *
 * @author dov
 */
public class ISingleton extends Singleton implements IArray{

    /**
     * The first element in the proffered array is this singleton.
     * @param array An array from which the first element becomes this singleton.
     * @param index The index of the desired entry.
     */
    public ISingleton(IArray array, int index) {
        super(array, index);
    }

    /**
     * Creates an empty singleton. 
     */
    public ISingleton() {
        super(Sizeof.INT);
    }
    
    
    
    /**
     * Gets the element in this singleton.
     * @param handle
     * @return The element in this singleton.
     */
    public int getVal(Handle handle){
        int[] cpuArray = new int[1];
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
    public ISingleton set(Handle handle, int d){
        set(handle, P.to(d));
        return this;
    }
    
    
    /**
     * {@inheritDoc}
     */
    @Override
    public ISingleton get(int i){
        return (ISingleton) super.get(i);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public int[] get(Handle handle) {
        int[] cpuArray = new int[1];
        get(handle, Pointer.to(cpuArray));
        return cpuArray;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public ISingleton copy(Handle handle) {
        return new ISingleton().set(handle, this);
    }
    
    
    /**
     * {@inheritDoc}
     */
    public ISingleton set(Handle handle, IArray from) {
        super.set(handle, from);
        return this;
    }
    
}
