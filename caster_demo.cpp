#include <iostream>
using std::cerr;
using std::cout;

#include <fstream>
using std::ofstream;

#include "caster.h"

int main(/*int argc, char const *argv[]*/)
{
    constexpr dimension width = 512;
    constexpr dimension height = 512;
    constexpr scalar fov = 60.0_deg;
    constexpr scalar focal_length = 0.25;

    Caster caster("maps/demo_map.dat", "textures/textures128x128.ppm", width, height, fov, focal_length);

    if (!caster.ready)
    {
        cerr << "Could not create caster.\n";
        return EXIT_FAILURE;
    }

    cout << "Map is " << caster.map_xsize << "x" << caster.map_ysize << " (" << caster.map.size() << ")\n";
    cout << "Camera is at " << caster.camx << ", " << caster.camy << " : " << caster.cama << " rads\n";

    // for (size_t row = 0; row < caster.map_ysize; row++)
    // {
    //     for (size_t col = 0; col < caster.map_xsize; col++)
    //     {
    //         char c = caster.map[col + row * caster.map_xsize] == 0 ? 'X' : '0';
    //         cout << c << " ";
    //     }
    //     cout << "\n";
    // }

    caster.draw();

    ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << caster.can_xsize << ' ' << caster.can_ysize << '\n'
        << "255\n";
    ofs.write(reinterpret_cast<char *>(caster.canvas.data()), caster.can_xsize * caster.can_ysize * sizeof(caster.canvas[0]));

    return EXIT_SUCCESS;
}
