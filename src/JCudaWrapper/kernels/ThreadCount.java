package JCudaWrapper.kernels;

/**
 *
 * @author E. Dov Neimand
 */
public class ThreadCount {

    public final static int threadsPerBlock1d = 256;
    public final static int threadsPerBlock2d = 16;
    public final static int threadsPerBlock3d = 8;

    private final int threadsPerBlock;

    public final int x, y, z;

    /**
     * The number of threads.
     *
     * @param x
     */
    public ThreadCount(int x) {
        this.x = x;
        this.y = this.z = 1;
        threadsPerBlock = threadsPerBlock1d;
    }

    public ThreadCount(int x, int y) {
        this.x = x;
        this.y = y;
        this.z = 1;
        threadsPerBlock = threadsPerBlock2d;
    }

    public ThreadCount(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
        threadsPerBlock = threadsPerBlock3d;
    }

    public ThreadCount(int x, int y, int z, int t) {
        this.x = x * t;
        this.y = y;
        this.z = z;
        threadsPerBlock = threadsPerBlock3d;
    }

    /**
     * The numver of blocks for the given number of threads.
     *
     * @param numThreads The number of threads.
     * @return The number of blocks.
     */
    protected int blocksPerGrid(int numThreads) {
        return ((int) Math.ceil((double) numThreads / threadsPerBlock));
    }

    /**
     * The number of blocks in the x direction.
     *
     * @return
     */
    public int xBlocksPerGrid() {
        return blocksPerGrid(x);
    }

    /**
     * The number of blocks in the y direction.
     *
     * @return
     */
    public int yBlocksPerGrid() {
        return blocksPerGrid(y);
    }

    /**
     * The number of blocks in the z direction.
     *
     * @return
     */
    public int zBlocksPerGrid() {
        return blocksPerGrid(z);
    }

    /**
     * The number of threads per block.
     *
     * @return
     */
    public int xThreadsPerBlock() {
        return threadsPerBlock;
    }

    /**
     * The number of threads per block.
     *
     * @return
     */
    public int yThreadsPerBlock() {
        return threadsPerBlock == threadsPerBlock1d ? 1 : threadsPerBlock;
    }

    /**
     * The number of threads per block.
     *
     * @return
     */
    public int zThreadsPerBlock() {
        return threadsPerBlock > threadsPerBlock3d ? 1 : threadsPerBlock;
    }

    @Override
    public String toString() {
        return String.format(
                "CUDA Thread Configuration:\n"
                + "--------------------------\n"
                + "Target Dimensions: [x:%d, y:%d, z:%d]\n"
                + "Threads Per Block: [x:%d, y:%d, z:%d] (Total: %d)\n"
                + "Blocks Per Grid:   [x:%d, y:%d, z:%d]\n"
                + "Total Threads:     [x:%d, y:%d, z:%d]",
                x, y, z,
                xThreadsPerBlock(), yThreadsPerBlock(), zThreadsPerBlock(),
                (xThreadsPerBlock() * yThreadsPerBlock() * zThreadsPerBlock()),
                xBlocksPerGrid(), yBlocksPerGrid(), zBlocksPerGrid(),
                (xBlocksPerGrid() * xThreadsPerBlock()),
                (yBlocksPerGrid() * yThreadsPerBlock()),
                (zBlocksPerGrid() * zThreadsPerBlock())
        );
    }
}
