#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../caster/caster2.h"

namespace py = pybind11;

PYBIND11_MODULE(caster2bind, m)
{
    py::class_<Caster>(m, "Caster", py::buffer_protocol())
        .def(py::init<>())
        .def_readwrite("buffer", &Caster::buffer);

    // .def("get", [](const Caster &cast) { return cast.buffer.data(); })
    // .def_buffer([](Caster &m) -> py::buffer_info {
    //     return py::buffer_info(
    //         m.buffer.data(),                         /* Pointer to buffer */
    //         sizeof(m.buffer[0]),                     /* Size of one scalar */
    //         py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
    //         1,                                       /* Number of dimensions */
    //         {screenWidth * screenHeight},            /* Buffer dimensions */
    //         {
    //             sizeof(m.buffer[0]) * screenWidth /* Strides (in bytes) for each index */
    //         });
    // });
    ;
}
