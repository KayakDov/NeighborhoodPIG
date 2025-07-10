package FijiInput;

import java.nio.file.Path;
import java.nio.file.Paths; // Added import for Paths, needed by Path.of()
import java.util.Optional;

/**
 * Abstract base for parsing a string argument into a desired type.
 * <p>
 * This class provides the string to parse and pre-composes standardized success
 * and error messages. Subclasses are responsible for implementing the concrete
 * parsing logic in the {@link #parse()} method.
 * </p>
 *
 * @param <T> The target type for the parsed argument.
 * @author E. Dov Neimand
 */
public abstract class Parser<T> {

    /**
     * The string value to be parsed.
     */
    protected final String toParse;

    /**
     * Pre-composed message for parsing failures.
     */
    protected final String errorMessage;

    /**
     * Pre-composed message for successful parsing.
     */
    protected final String successMessage;

    /**
     * True if the returned result should not be empty. If this flag is true and
     * {@link #opt()} is called, it will immediately return
     * {@link Optional#empty()} without attempting to parse.
     */
    boolean present;

    /**
     * Constructs a {@code Parser} instance. Initializes the string to parse and
     * pre-composes messages.
     *
     * @param toParse The string value to be parsed.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "an integer").
     * @param present If false, {@link #opt()} will immediately return
     * {@link Optional#empty()} without attempting to parse. This is typically
     * used when the input string itself signifies an absent optional value
     * (e.g., an empty string or a specific keyword like "false"). Note: The
     * field name `present` (was `present`) suggests the opposite meaning of
     * this parameter's current usage in `opt()`.
     */
    public Parser(StringIter toParse, String expected, boolean present) {
        int index = toParse.getIndex();
        try {
            this.toParse = toParse.next(present);
            String preamble = "At index " + index + ", ";
            this.errorMessage = preamble + "'" + this.toParse + "' could not be parsed. '" + expected + "' was expected.";
            this.successMessage = preamble + "successfully parsed " + expected + " = '" + this.toParse + "'.";
            this.present = present;
        } catch (ArrayIndexOutOfBoundsException aioobe) {
            throw new ArrayIndexOutOfBoundsException("You didn't pass enough paramaters.  The next paramater should be '" + expected + "'.");
        }

    }

    /**
     * Abstract method for concrete subclasses to implement the actual parsing
     * logic. This method attempts to convert {@link #toParse} into type
     * {@code T}.
     *
     * @return The parsed value of type {@code T}. Returns a non-null value upon
     * successful parsing.
     * @throws IllegalArgumentException If {@link #toParse} cannot be converted
     * into type {@code T} due to a format error or invalid data.
     * @throws NumberFormatException (Implicit) If the parsing logic within the
     * subclass involves numeric conversion and the string format is invalid.
     */
    protected abstract T parse() throws IllegalArgumentException;

    /**
     * Attempts to parse the argument and returns an {@link Optional} containing
     * the result.
     * <p>
     * If the {@link #present} (originally `present`) flag, set in the
     * constructor, is false, this method immediately returns
     * {@link Optional#empty()} without attempting to parse the {@link #toParse}
     * string.
     * </p>
     * <p>
     * If the flag is true, it attempts to parse the string using
     * {@link #parse()}. On successful parsing, it prints a success message to
     * standard output.
     * </p>
     *
     * @return An {@link Optional} containing the parsed value if successful and
     * {@link #present} is true. Returns {@link Optional#empty()} if
     * {@link #present} is false.
     * @throws NumberFormatException If {@link #parse()} throws a
     * {@code NumberFormatException}. The exception's message will be replaced
     * with {@link #errorMessage}.
     * @throws IllegalArgumentException If {@link #parse()} throws an
     * {@code IllegalArgumentException}. The exception's message will be
     * replaced with {@link #errorMessage}.
     */
    public Optional<T> opt() throws NumberFormatException, IllegalArgumentException {
        if (!present) return Optional.empty(); // This logic means 'present' (which was 'present') is effectively 'should be empty' if false.
        // This is a bit counter-intuitive for a 'present' flag.
        try {
            T val = parse();
            System.out.println("FijiInput.Parser.get() " + successMessage); // Original System.out.println still uses .get() in string
            return Optional.of(val); // This will throw NullPointerException if parse() returns null.
        } catch (NumberFormatException nfe) {
            throw new NumberFormatException(errorMessage);
        } catch (IllegalArgumentException iae) {
            throw new IllegalArgumentException(errorMessage, iae);
        }
    }

    /**
     * Returns the pre-composed error message for this parser instance.
     *
     * @return The error message string.
     */
    public String getErrorMessage() {
        return errorMessage;
    }

    /**
     * Returns the pre-composed success message for this parser instance.
     *
     * @return The success message string.
     */
    public String getSuccessMessage() {
        return successMessage;
    }
}

// --- Concrete Parser Implementations ---
/**
 * Parses a string to a Boolean.
 */
class ParseBool extends Parser<Boolean> {

    /**
     * Constructs a ParseBool instance.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "a boolean").
     * @param present If true, indicates the parser should attempt to parse the
     * string; if false, {@link #opt()} will immediately return
     * {@link Optional#empty()}.
     */
    public ParseBool(StringIter toParse, String expected, boolean present) {
        super(toParse, expected, present);
    }

    /**
     * Constructs a ParseBool instance where the boolean value is expected to be
     * present. The `present` parameter for the superclass is defaulted to
     * `true`.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "a boolean").
     */
    public ParseBool(StringIter toParse, String expected) {
        super(toParse, expected, true);
    }

    /**
     * Parses the string to a Boolean.
     *
     * @return {@code true} if {@link #toParse} is "true" (case-insensitive),
     * {@code false} if "false" (case-insensitive).
     * @throws NumberFormatException If {@link #toParse} is not "true" or
     * "false" (case-insensitive). (Note: The parent's `opt()` method will
     * replace this exception's message with `errorMessage`).
     */
    @Override
    protected Boolean parse() throws IllegalArgumentException {
        if (toParse.equalsIgnoreCase("true")) return true;
        if (toParse.equalsIgnoreCase("false")) return false;
        throw new NumberFormatException(errorMessage);
    }

    /**
     * Convenience static method to parse a string to an optional Boolean.
     * Creates a new {@code ParseBool} instance and calls its {@link #opt()}
     * method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "a
     * boolean").
     * @param present If true, indicates a value is expected to be parsed; if
     * false, {@link Optional#empty()} is returned directly without parsing.
     * @return An {@link Optional} containing the parsed Boolean, or
     * {@link Optional#empty()} if `present` is false.
     * @throws NumberFormatException If `toParse` is not "true" or "false" and
     * `present` is true.
     * @throws IllegalArgumentException If an unexpected parsing error occurs
     * and `present` is true.
     */
    public static Optional<Boolean> from(StringIter toParse, String expected, boolean present) {
        return new ParseBool(toParse, expected, present).opt();
    }

    /**
     * Convenience static method to parse a string to a present Boolean
     * (defaulting `present` to true). Creates a new {@code ParseBool} instance
     * and calls its {@link #opt()} method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "a
     * boolean").
     * @return An {@link Optional} containing the parsed Boolean.
     * @throws NumberFormatException If `toParse` is not "true" or "false".
     * @throws IllegalArgumentException If an unexpected parsing error occurs.
     */
    public static Optional<Boolean> from(StringIter toParse, String expected) {
        return new ParseBool(toParse, expected, true).opt();
    }
}

/**
 * Parses a string to an Integer.
 */
class ParseInt extends Parser<Integer> {

    /**
     * Constructs a ParseInt instance.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "an integer").
     * @param present If true, indicates the parser should attempt to parse the
     * string; if false, {@link #opt()} will immediately return
     * {@link Optional#empty()}.
     */
    public ParseInt(StringIter toParse, String expected, boolean present) {
        super(toParse, expected, present);
    }

    /**
     * Constructs a ParseInt instance where the integer value is expected to be
     * present. The `present` parameter for the superclass is defaulted to
     * `true`.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "an integer").
     */
    public ParseInt(StringIter toParse, String expected) {
        super(toParse, expected, true);
    }

    /**
     * {@inheritDoc }
     * Parses the string to an Integer.
     *
     * @return The parsed {@code Integer} value.
     * @throws NumberFormatException If {@link #toParse} cannot be parsed into
     * an integer. (Note: The parent's `opt()` method will replace this
     * exception's message with `errorMessage`).
     */
    @Override
    protected Integer parse() throws IllegalArgumentException {
        return Integer.valueOf(toParse);
    }

    /**
     * Convenience static method to parse a string to an optional Integer.
     * Creates a new {@code ParseInt} instance and calls its {@link #opt()}
     * method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "an
     * integer").
     * @param present If true, indicates a value is expected to be parsed; if
     * false, {@link Optional#empty()} is returned directly without parsing.
     * @return An {@link Optional} containing the parsed Integer, or
     * {@link Optional#empty()} if `present` is false.
     * @throws NumberFormatException If `toParse` cannot be parsed to an Integer
     * and `present` is true.
     * @throws IllegalArgumentException If an unexpected parsing error occurs
     * and `present` is true.
     */
    public static Optional<Integer> from(StringIter toParse, String expected, boolean present) {
        return new ParseInt(toParse, expected, present).opt();
    }

    /**
     * Convenience static method to parse a string to a present Integer
     * (defaulting `present` to true). Creates a new {@code ParseInt} instance
     * and calls its {@link #opt()} method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "an
     * integer").
     * @return An {@link Optional} containing the parsed Integer.
     * @throws NumberFormatException If `toParse` cannot be parsed to an
     * Integer.
     * @throws IllegalArgumentException If an unexpected parsing error occurs.
     */
    public static Optional<Integer> from(StringIter toParse, String expected) {
        return new ParseInt(toParse, expected, true).opt();
    }
}

/**
 * Parses a string to a Double.
 */
class ParseReal extends Parser<Double> {

    /**
     * Constructs a ParseDouble instance.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "a decimal number").
     * @param present If true, indicates the parser should attempt to parse the
     * string; if false, {@link #opt()} will immediately return
     * {@link Optional#empty()}.
     */
    public ParseReal(StringIter toParse, String expected, boolean present) {
        super(toParse, expected, present);
    }

    /**
     * Constructs a ParseDouble instance where the double value is expected to
     * be present. The `present` parameter for the superclass is defaulted to
     * `true`.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected argument type
     * (e.g., "a decimal number").
     */
    public ParseReal(StringIter toParse, String expected) {
        super(toParse, expected, true);
    }

    /**
     * {@inheritDoc }
     * Parses the string to a Double.
     *
     * @return The parsed {@code Double} value.
     * @throws NumberFormatException If {@link #toParse} cannot be parsed into a
     * double. (Note: The parent's `opt()` method will replace this exception's
     * message with `errorMessage`).
     */
    @Override
    protected Double parse() throws IllegalArgumentException {
        return Double.valueOf(toParse);
    }

    /**
     * Convenience static method to parse a string to an optional Double.
     * Creates a new {@code ParseDouble} instance and calls its {@link #opt()}
     * method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "a
     * real number").
     * @param present If true, indicates a value is expected to be parsed; if
     * false, {@link Optional#empty()} is returned directly without parsing.
     * @return An {@link Optional} containing the parsed Double, or
     * {@link Optional#empty()} if `present` is false.
     * @throws NumberFormatException If `toParse` cannot be parsed to a Double
     * and `present` is true.
     * @throws IllegalArgumentException If an unexpected parsing error occurs
     * and `present` is true.
     */
    public static Optional<Double> from(StringIter toParse, String expected, boolean present) {
        return new ParseReal(toParse, expected, present).opt();
    }

    /**
     * Convenience static method to parse a string to a present Double
     * (defaulting `present` to true). Creates a new {@code ParseDouble}
     * instance and calls its {@link #opt()} method.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "a
     * real number").
     * @return An {@link Optional} containing the parsed Double.
     * @throws NumberFormatException If `toParse` cannot be parsed to a Double.
     * @throws IllegalArgumentException If an unexpected parsing error occurs.
     */
    public static Optional<Double> from(StringIter toParse, String expected) {
        return new ParseReal(toParse, expected, true).opt();
    }
}

/**
 * Parses a string to a Path.
 */
class ParsePath extends Parser<Path> {

    /**
     * Constructs a ParsePath instance. The `present` flag for the superclass is
     * determined by whether `toParse` is equal to "false" (case-insensitive).
     * If "false", `present` is set to `false`, meaning {@link #opt()} will
     * return {@link Optional#empty()}. Otherwise, it's `true`.
     *
     * @param toParse The string to parse, representing a file path or the
     * literal "false".
     * @param expected A descriptive string for the expected argument type
     * (e.g., "a file path").
     */
    public ParsePath(StringIter toParse, String expected) {
        super(toParse, expected, !toParse.peek().equalsIgnoreCase("false"));
        if (toParse.peek().equalsIgnoreCase("false")) toParse.next();
    }

    /**
     * {@inheritDoc }
     * Parses the string to a {@link Path} object.
     *
     * @return The parsed {@link Path} value.
     * @throws IllegalArgumentException If {@link #toParse} cannot be converted
     * into a valid {@link Path}.
     */
    @Override
    protected Path parse() throws IllegalArgumentException {
        // Path.of(String) can throw InvalidPathException, which is a RuntimeException
        // and a subclass of IllegalArgumentException.
        return Paths.get(toParse); // Changed to Paths.get(toParse) for clarity and typical usage with Path.of()
    }

    /**
     * Convenience static method to parse a string to a Path. Creates a new
     * {@code ParsePath} instance and calls its {@link #opt()} method. The
     * `present` flag is determined by the `ParsePath` constructor itself,
     * allowing "false" as a valid input for an empty optional.
     *
     * @param toParse The string to parse.
     * @param expected A descriptive string for the expected value (e.g., "a
     * file path").
     * @return An {@link Optional} containing the parsed Path, or
     * {@link Optional#empty()} if `toParse` is "false" (case-insensitive).
     * @throws IllegalArgumentException If `toParse` is not "false" and cannot
     * be parsed to a Path.
     */
    public static Optional<Path> from(StringIter toParse, String expected) {

        return new ParsePath(toParse, expected).opt();
    }

}
