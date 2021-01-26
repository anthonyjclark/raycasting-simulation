#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../RaycastWorld/RaycastWorld.h"

namespace py = pybind11;

PYBIND11_MODULE(pycaster, m)
{
    py::enum_<Turn>(m, "Turn", py::arithmetic(), "Turning enumeration")
        .value("Left", LEFT)
        .value("Stop", STOP)
        .value("Right", RIGHT);

    py::enum_<Walk>(m, "Walk", py::arithmetic(), "Walking enumeration")
        .value("Backward", BACKWARD)
        .value("Stopped", STOPPED)
        .value("Forward", FORWARD);

    py::class_<RaycastWorld>(m, "RaycastWorld", py::buffer_protocol())
        .def(py::init<usize, usize, std::string>())
        .def("turn", &RaycastWorld::setTurn)
        .def("walk", &RaycastWorld::setWalk)
        .def("update", &RaycastWorld::updatePose)
        .def("render", &RaycastWorld::renderView)
        .def("getX", &RaycastWorld::getX)
        .def("getY", &RaycastWorld::getY)
        .def("getDirX", &RaycastWorld::getDirX)
        .def("getDirY", &RaycastWorld::getDirY)
        .def("position", &RaycastWorld::setPosition)
        .def("direction", &RaycastWorld::setDirection)
        .def_buffer([](RaycastWorld &caster) -> py::buffer_info {
            return py::buffer_info(
                caster.getBuffer(),                       // Pointer to data
                sizeof(uint8_t),                          // Size of data type
                py::format_descriptor<uint8_t>::format(), // Data type for messages
                3,                                        // Number of dimensions
                {
                    static_cast<int>(caster.getHeight()), // Height of buffer
                    static_cast<int>(caster.getWidth()),  // Width of buffer
                    3                                     // Number of color channels
                },
                {
                    sizeof(uint8_t) * caster.getWidth() * 3, // Stride for height
                    sizeof(uint8_t) * 3,                     // Stride for width
                    sizeof(uint8_t)                          // Stride for pixels
                });
        });
}
