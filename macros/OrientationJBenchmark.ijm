// ─────────────────── USER‑ADJUSTABLE SETTINGS ───────────────────
tensor     = 15;              // OrientationJ tensor parameter
increment  = 5;              // How many extra slices per iteration (5 → 5,10,15…)
start      = 530;
inputDir   = "/home/dov/projects/NieghborhoodPIG/images/input/misize/";
logFile    = "/home/dov/projects/NieghborhoodPIG/orientationJ_times.txt";
// ────────────────────────────────────────────────────────────────

// Get total number of images in the folder
fileList = getFileList(inputDir);
total    = fileList.length;

for (size = start; size <= total; size += increment) {

    // ── OPEN A STACK OF THE FIRST <size> IMAGES ──
    run("Image Sequence...", 
        "open=[" + inputDir + "] sort " + 
        "number=" + size + " starting=1 increment=1 scale=100");
    stackTitle  = getTitle(); // Title of the just‑opened stack
    sliceCount  = nSlices;    // Should equal 'size'
    run("Stack to Hyperstack...", "order=xyczt(default) channels=1 slices=1 frames=" + size + " display=Color");

    // ── RUN ORIENTATIONJ AND TIME IT ──
    startTime = getTime();
//    15 true false false 1
 run("Neighborhood PIG", tensor +" true false false 1");
//    run("OrientationJ Analysis",
//        "tensor=" + tensor +
//        " gradient=0 hsb=on hue=Orientation sat=Coherency " +
//        "bri=Original-Image orientation=on radian=on");

    elapsed = getTime() - startTime;

    // ── CLEAN‑UP ──
    run("Close All");
    run("Collect Garbage");        // Free JVM heap between iterations

    File.append(tensor + ", " + sliceCount + ", " + elapsed, logFile);

    print("Finished stack of " + sliceCount + " slices → " + elapsed + " ms");
}

