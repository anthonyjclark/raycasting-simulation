#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../caster/caster2.h"

namespace py = pybind11;

PYBIND11_MODULE(pycaster, m)
{
    py::class_<Caster>(m, "Caster", py::buffer_protocol())
        .def(py::init<uint32_t, uint32_t>())
        .def("render", &Caster::render)
        .def_buffer([](Caster &caster) -> py::buffer_info {
            return py::buffer_info(
                caster.getBuffer(),                       // Pointer to data
                sizeof(uint8_t),                          // Size of data type
                py::format_descriptor<uint8_t>::format(), // Data type for messages
                3,                                        // Number of dimensions
                {
                    static_cast<int>(caster.height()), // Height of buffer
                    static_cast<int>(caster.width()),  // Width of buffer
                    3                                  // Number of color channels
                },
                {
                    sizeof(uint8_t) * caster.width() * 3, // Stride for height
                    sizeof(uint8_t) * 3,                  // Stride for width
                    sizeof(uint8_t)                       // Stride for pixels
                });
        });
}
