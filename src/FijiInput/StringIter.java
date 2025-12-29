package FijiInput;

import java.util.Iterator;

/**
 * A class to help with counting.
 *
 * @author E. Dov Neimand
 */
public class StringIter implements Iterator<String> {

    private int i;
    private String[] args;

    /**
     * Creates teh iterator.
     *
     * @param args The strings to be iteratoed over.
     */
    public StringIter(String[] args) {
        i = 0;
        this.args = args;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public String next() {
        return args[i++];
    }

    /**
     * The next String if the condition is met, otherwise null.
     *
     * @param condition If this is true, the next string.
     */
    public String next(boolean condition) {
        if (condition) return next();
        else return null;
    }

    /**
     * {@inheritDoc }
     */
    @Override
    public boolean hasNext() {
        return i < args.length;
    }

    /**
     * The next element is revealed, but the iterator remains at the current
     * location.
     *
     * @return The next element is revealed, but the iterator remains at the
     * current location.
     */
    public String peek() {
        return args[i];
    }

    public int getIndex() {
        return i;
    }
    
    
}
