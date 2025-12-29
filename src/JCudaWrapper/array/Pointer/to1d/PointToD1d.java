package JCudaWrapper.array.Pointer.to1d;

import JCudaWrapper.array.Double.DArray1d;
import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Pointer.to2d.PointTo2d;
import JCudaWrapper.resourceManagement.Handle;

/**
 *
 * @author E. Dov Neimand
 */
public interface PointToD1d extends PointTo1d{
    /**
     * {@inheritDoc }
     * TODO: can this be done on the gpu?  IMplement in To2d instead of ToD2d?
     */
    @Override
    public default PointTo1d initTargets(Handle hand){
        for(int i = 0; i < size(); i++)
            get(i).set(hand, new DArray1d(size()));
        return this;
    }
}
