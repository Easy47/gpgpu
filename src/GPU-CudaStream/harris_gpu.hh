#pragma once
#include <cstddef>
#include <memory>
#include "../png/png_handler.hh"


namespace gpuCudaStream {
	typedef struct
	{
		int x, y;
	} Point;

	void detect_point(PNG_data image_data);
}

