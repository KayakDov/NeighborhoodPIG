package fijiPlugin;

import java.io.*;
import java.net.URI;
import java.nio.file.*;
import java.util.jar.JarFile;
import java.lang.reflect.Method;
import java.net.URLClassLoader;

/**
 * Ensures JCuda and JCublas libraries are available in Fiji's jars folder.
 * If missing, downloads and dynamically loads them.
 */
public class InitJCuda {
    
    private static final String JARS_DIR = "../jars/";
    private static final String VERSION = "12.6.0";
    private static final String BASE_URL = "https://repo1.maven.org/maven2/org/jcuda/";
    private static final String[] LIBRARIES = new String[]{"jcuda", "jcuda-natives", "jcublas", "jcublas-natives"};
    
    public static void main(String[] args) {
        ensureLibraries();
    }
    
    /**
     * Ensures all required JCuda libraries are present and loads them.
     */
    public static void ensureLibraries() {
        String os = detectOS();
        String arch = detectArch();
        
        for (String lib : LIBRARIES) {
            String jarPath = JARS_DIR + lib + "-" + VERSION + ".jar";
            if (!isJarValid(jarPath)) downloadJar(lib, jarPath, os, arch);
            loadJar(jarPath);
        }
    }
    
    /**
     * Detects the operating system.
     */
    private static String detectOS() {
        String osName = System.getProperty("os.name").toLowerCase();
        if (osName.contains("win")) return "windows";
        if (osName.contains("mac")) return "macos";
        if (osName.contains("nux")) return "linux";
        throw new RuntimeException("Unsupported OS: " + osName);
    }
    
    /**
     * Detects the system architecture.
     */
    private static String detectArch() {
        String arch = System.getProperty("os.arch");
        if (arch.contains("64")) return "x86_64";
        if (arch.contains("86")) return "x86";
        throw new RuntimeException("Unsupported architecture: " + arch);
    }
    
    /**
     * Checks if a JAR file exists and is valid.
     */
    private static boolean isJarValid(String jarPath) {
        File file = new File(jarPath);
        return file.exists() && isValidJar(file);
    }
    
    /**
     * Validates whether the given file is a proper JAR file.
     */
    private static boolean isValidJar(File file) {
        try (JarFile jar = new JarFile(file)) {
            return true;
        } catch (IOException e) {
            return false;
        }
    }
    
    /**
     * Downloads a JAR from the Maven repository, checking OS and architecture.
     */
    private static void downloadJar(String lib, String jarPath, String os, String arch) {
        String url = BASE_URL + lib + "/" + VERSION + "/" + lib + "-" + VERSION + "-" + os + "-" + arch + ".jar";
        try (InputStream in = URI.create(url).toURL().openStream()) {
            Files.copy(in, Paths.get(jarPath), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new RuntimeException("Failed to download " + lib, e);
        }
    }
    
    /**
     * Dynamically loads a JAR file into the system class loader.
     */
    private static void loadJar(String jarPath) {
        try {
            File jarFile = new File(jarPath);
            URI jarUri = jarFile.toURI();
            URLClassLoader classLoader = (URLClassLoader) ClassLoader.getSystemClassLoader();
            Method method = URLClassLoader.class.getDeclaredMethod("addURL", URI.class);
            method.setAccessible(true);
            method.invoke(classLoader, jarUri.toURL());
        } catch (Exception e) {
            throw new RuntimeException("Failed to load " + jarPath, e);
        }
    }
}
