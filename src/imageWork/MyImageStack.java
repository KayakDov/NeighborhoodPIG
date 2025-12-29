package imageWork;

import ij.ImageStack;

/**
 *
 * @author dov
 */
public class MyImageStack extends ImageStack {

    /**
     * Constructs an empty image stack.
     *
     * @param width The width of the stack;
     * @param height The height of the stack.
     */
    public MyImageStack(int width, int height) {
        super(width, height);
    }

    /**
     * Concatenates the new stack onto this one.
     *
     * @param is The stack to be concatenated to this one.
     * @return this.
     */
    public MyImageStack concat(ImageStack is) {
                
        for (int i = 1; i <= is.getSize(); i++)
            addSlice(is.getSliceLabel(i), is.getProcessor(i));
        return this;
    }

    /**
     * An image plus from this.
     *
     * @param title The title of the image plus.
     * @param depth The depth of the new ImagePlus.
     * @return An ImagePlus object generated from this stack.
     */
    public MyImagePlus getImagePlus(String title, int depth) {
        return new MyImagePlus(title, this, depth);
    }


}
