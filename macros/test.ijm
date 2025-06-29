print("Running correct macro version");


// --- 1. Define the image directory (must be an absolute path) ---
imageDirectory = "/home/dov/projects/NieghborhoodPIG/images/input/cyl";

// --- 2. Load image sequence ---
run("Image Sequence...", "open=[" + imageDirectory + "/] sort");

if (nImages() == 0) {
    showMessage("Image Not Found", "No image was opened.");
    exit();
}

// --- 3. Set hyperstack dimensions ---
nChannels = 1;
nSlices = 50;
nFrames = 3;

stackSize = nSlices(); // total slices in stack
expectedSize = nChannels * nSlices * nFrames;

if (stackSize != expectedSize) {
    showMessage("Dimension Mismatch", 
        "Imported stack has " + stackSize + " slices, expected " + expectedSize);
    exit();
}

run("Properties...", "channels=" + nChannels +
                      " slices=" + nSlices +
                      " frames=" + nFrames +
                      " unit=pixel pixel_width=1.0 pixel_height=1.0 voxel_depth=1.0 frame_interval=1.0");


/*// --- 4. Run Neighborhood PIG Plugin ---
xyR = 5;
zR = 5;
zSpacingMultiplier = 1.0;
generateHeatmap = true;
generateVectorField = false;
generateCoherence = true;
downsampleFactor = 1;

params = xyR + " " +
         zR + " " +
         zSpacingMultiplier + " " +
         generateHeatmap + " " +
         generateVectorField + " " +
         generateCoherence + " " +
         downsampleFactor;

print("Running Neighborhood PIG with parameters: " + params);
run("Neighborhood PIG", params);
print("Macro finished.");
*/
