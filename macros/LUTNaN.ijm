// Assumes current image is 32-bit float with NaNs and values in [0, π]
originalTitle = getTitle();
getDimensions(width, height, channels, slices, frames);

// Create RGB image
newImage("ColorMapped", "RGB black", width, height, 1);
resultTitle = "ColorMapped";

// Allocate RGB array for pixel setting
rgb = newArray(3);

// Loop through all pixels
for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
        selectWindow(originalTitle);
        value = getPixel(x, y);

        if (isNaN(value)) {
            rgb[0] = 0; rgb[1] = 0; rgb[2] = 0;
        } else {
            hue = value / PI;
            hsvToRgb(hue, 1, 1); // sets global r, g, b
            rgb[0] = r; rgb[1] = g; rgb[2] = b;
        }

        selectWindow(resultTitle);
        setPixel(x, y, rgb); // Set RGB directly
    }
}

rename("Phase → HSV Color (NaN=Black)");

// --- HSV to RGB conversion (sets r, g, b)
function hsvToRgb(h, s, v) {
    h = 6 * h;
    i = floor(h);
    f = h - i;
    p = 255 * v * (1 - s);
    q = 255 * v * (1 - s * f);
    t = 255 * v * (1 - s * (1 - f));
    v = 255 * v;

    if (i == 0)      { r = v; g = t; b = p; }
    else if (i == 1) { r = q; g = v; b = p; }
    else if (i == 2) { r = p; g = v; b = t; }
    else if (i == 3) { r = p; g = q; b = v; }
    else if (i == 4) { r = t; g = p; b = v; }
    else             { r = v; g = p; b = q; }

    r = floor(r); g = floor(g); b = floor(b);
}

