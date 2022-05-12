#pragma once
#include <iostream>
#include <png.h>

struct PNG_data {
    int width;
    int height;
    png_byte color_type;
    png_byte bit_depth;
    png_bytep *row_pointers = NULL;
};

struct PNG_data read_png_file(const char *filename);
