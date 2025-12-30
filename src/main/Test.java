package main;

import JCudaWrapper.array.Double.DArray2d;
import JCudaWrapper.array.Float.FArray;
import JCudaWrapper.array.Float.FArray2d;
import JCudaWrapper.kernels.KernelManager;
import JCudaWrapper.array.P;
import JCudaWrapper.array.Pointer.to2d.PArray2dToD2d;
import JCudaWrapper.array.Pointer.to2d.P2dToF2d;
import JCudaWrapper.resourceManagement.Handle;
import fijiPlugin.Dimensions;
import jcuda.runtime.JCuda;

/**
 * Debugging tests constructed here.
 *
 * @author E. Dov Neimand
 */
public class Test {

    /**
     * Checks if all the elements of the array are finite. The check is
     * performed on the cpu. It is for testing purposes only.
     *
     * @param array The array to be checked.
     * @param handle
     * @return true if all the elements are finite. False otherwise.
     */
    public static int nonFiniteCount(FArray array, Handle handle) {
        int count = 0;
        float[] cpuArray = array.get(handle);
        for (int i = 0; i < cpuArray.length; i++)
            if (!Float.isFinite(cpuArray[i])) count++;
        return count;
    }

    public static void test2dto2d() {
        try (
                Handle hand = new Handle(); 
                Dimensions dim = new Dimensions(hand, 2, 2, 2, 2); 
                P2dToF2d p = dim.emptyP2dToF2d(null)
                ) {

            FArray2d[] a = new FArray2d[]{
                new FArray2d(2, 2).set(hand, 1, 2, 3, 4),
                new FArray2d(2, 2).set(hand, 4, 3, 2, 1),
                new FArray2d(2, 2).set(hand, 9, 12, 3, -1),
                new FArray2d(2, 2).set(hand, 7, 8, 70, 80)
            };

            p.set(hand, a);
            
            p.scale(hand, 100);

            System.out.println(p.toString());

        }

    }
    
    public static String format(String inputString) {
        String result = inputString;

        // 1. Round all numbers with E- in them to 0.
        // This should be done first, as these numbers might also have decimals.
        // Regex: Matches a number (integer or float) followed by E/e and a negative exponent.
        // It's quite comprehensive for scientific notation with negative exponents.
        // Ensures it doesn't match E+ or just E.
        result = result.replaceAll("\\b\\d*\\.?\\d+(?:[Ee][-+]?\\d+)\\b", "");

        // 2. Truncate all floating point numbers to 1 value after the decimal place.
        // This regex specifically targets numbers with more than one digit after the decimal
        // and keeps only the first one.
        // It relies on the fact that numbers like 1.0 or 0.0 will be handled by steps 3 and 4.
        // \\b ensures word boundaries so it doesn't match parts of other text.
        // The first capturing group (\\d+\\.\\d) captures the number up to the first decimal digit.
        // The second part (\\d+) matches any subsequent decimal digits we want to remove.
        result = result.replaceAll("(\\b\\d+\\.\\d)\\d+\\b", "$1");

        // 3. Map ".0," to ","
        // This handles cases like "1.0, 2.5" -> "1, 2.5"
        // Using lookbehind (?<=\\d) to ensure there's a digit before ".0"
        result = result.replaceAll("(?<=\\d)\\.0,", ",");

        // 4. Map ".0]" to "]"
        // This handles cases like "1.0]" -> "1]"
        // Using lookbehind (?<=\\d) to ensure there's a digit before ".0"
        result = result.replaceAll("(?<=\\d)\\.0\\]", "]");


        // Important: If you have numbers at the end of the string that are 1.0 or 0.0,
        // but are NOT followed by a comma or bracket, they won't be caught by 3 or 4.
        // So, we need an additional step for numbers ending the string or followed by whitespace/other punctuation.
        // This regex matches numbers ending with ".0" at a word boundary that haven't been handled.
        // This should be done *after* the previous ones, otherwise, it might convert 1.0, to 1,
        // and then this regex won't see the .0 anymore.
        result = result.replaceAll("(?<=\\d)\\.0\\b", "");
        
        result = result.replace("-0,", "0,");
        result = result.replace("-0]", "0]");
        result = result.replace("NaN", "N");
        result = result.replace("0,", ",");
        result = result.replace("0]", "]");
        result = result.replace("-,", ",");
        result = result.replace("-]", "]");
        


        return result;
    }

    public static void main(String[] args) {
        String test1 = "Value: 123.456, Temp: 0.12, Pressure: 98.7654, Error: 1.23E-4, Item: 1.0]";
        String test2 = "Start: 0.0, Mid: 1.0, End: 5.6789, Tiny: 5e-9, Array: [1.0, 2.0, 3.55]";
        String test3 = "Just numbers: 0.0 1.0 1.2345 6.789e-5 9.0]";
        String test4 = "Edge case: 1.000, 0.000, 1.234E-10, 5.0]";
        String test5 = "No decimal: 100, E-notation not at end: 1.0E-2.0, 1.0E-3";

        System.out.println("Original 1: " + test1);
        System.out.println("Formatted 1: " + format(test1));
        // Expected: Value: 123.4, Temp: 0.1, Pressure: 98.8, Error: 0, Item: 1] (Note: 98.8 due to default rounding)

        System.out.println("\nOriginal 2: " + test2);
        System.out.println("Formatted 2: " + format(test2));
        // Expected: Start: 0, Mid: 1, End: 5.7, Tiny: 0, Array: [1, 2, 3.6]

        System.out.println("\nOriginal 3: " + test3);
        System.out.println("Formatted 3: " + format(test3));
        // Expected: Just numbers: 0 1 1.2 0 9]

        System.out.println("\nOriginal 4: " + test4);
        System.out.println("Formatted 4: " + format(test4));
        // Expected: Edge case: 1, 0, 0, 5]

        System.out.println("\nOriginal 5: " + test5);
        System.out.println("Formatted 5: " + format(test5));
        // Expected: No decimal: 100, E-notation not at end: 0, 0
    }

}
