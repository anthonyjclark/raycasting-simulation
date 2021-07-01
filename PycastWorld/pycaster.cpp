#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "../RaycastWorld/RaycastWorld.h"

namespace py = pybind11;

PYBIND11_MODULE(pycaster, m)
{
    py::enum_<Turn>(m, "Turn", py::arithmetic(), "Turning enumeration")
        .value("Left", Turn::LEFT)
        .value("Stop", Turn::STOP)
        .value("Right", Turn::RIGHT);

    py::enum_<Walk>(m, "Walk", py::arithmetic(), "Walking enumeration")
        .value("Backward", Walk::BACKWARD)
        .value("Stop", Walk::STOP)
        .value("Forward", Walk::FORWARD);

    py::class_<RaycastWorld>(m, "PycastWorld", py::buffer_protocol())
        .def(py::init<usize, usize, std::string>())
        .def("update", &RaycastWorld::update_pose)
        .def("render", &RaycastWorld::render_view)
        .def("reset", &RaycastWorld::reset)
        .def("save_png", &RaycastWorld::save_png)
        .def("at_goal", &RaycastWorld::at_goal)
        .def("turn", &RaycastWorld::set_turn)
        .def("walk", &RaycastWorld::set_walk)
        .def("x", &RaycastWorld::get_x)
        .def("y", &RaycastWorld::get_y)
        .def("set_position", &RaycastWorld::set_position_xy)
        .def("direction", &RaycastWorld::get_direction)
        .def("set_direction", &RaycastWorld::set_direction)
        .def("turn_speed", &RaycastWorld::get_turn_speed)
        .def("walk_speed", &RaycastWorld::get_walk_speed)
        .def("get_dir_x", &RaycastWorld::get_dir_x)
        .def("get_dir_y", &RaycastWorld::get_dir_y)
        .def_buffer([](RaycastWorld &caster) -> py::buffer_info
                    {
                        return py::buffer_info(
                            caster.get_buffer(),                      // Pointer to data
                            sizeof(uint8_t),                          // Size of data type
                            py::format_descriptor<uint8_t>::format(), // Data type for messages
                            3,                                        // Number of dimensions
                            {
                                static_cast<int>(caster.get_screen_height()), // Height of buffer
                                static_cast<int>(caster.get_screen_width()),  // Width of buffer
                                3                                             // Number of color channels
                            },
                            {
                                sizeof(uint8_t) * caster.get_screen_width() * 3, // Stride for height
                                sizeof(uint8_t) * 3,                             // Stride for width
                                sizeof(uint8_t)                                  // Stride for pixels
                            });
                    });
}
