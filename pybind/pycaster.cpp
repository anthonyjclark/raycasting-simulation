#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../caster/caster2.h"

namespace py = pybind11;

PYBIND11_MODULE(pycaster, m)
{
    py::class_<Caster>(m, "Caster", py::buffer_protocol())
        .def(py::init<double, double>())
        .def_buffer([](Caster &caster) -> py::buffer_info {
            return py::buffer_info(
                caster.buffer.data(),                     // Pointer to data
                sizeof(uint8_t),                          // Size of data type
                py::format_descriptor<uint8_t>::format(), // Data type for messages
                3,                                        // Number of dimensions
                {screenHeight, screenWidth, 3},           // Size of each dimension
                {
                    sizeof(uint8_t) * screenWidth * 3, // Stride for height
                    sizeof(uint8_t) * 3,               // Stride for width
                    sizeof(uint8_t)                    // Stride for pixels
                });
        });
}
