#include <iostream>
using std::cout;

#include "doom3d.h"

int main(int argc, char const *argv[])
{
    Doom3D game;
    game.scr_resize(512, 512);
    game.draw();

    cout << "Map dimensions: " << game.mxs << ", " << game.mys << '\n'
         << "Canvas dimensions: " << game.sxs << ", " << game.sys << '\n';

    ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << game.sxs << ' ' << game.sys << '\n'
        << "255\n";
    ofs.write(reinterpret_cast<char *>(game.canvas), game.sys * game.sxs * 3);
    // << game.canvas << '\n';

    // for (auto j = 0u; j < game.sys; ++j)
    // {
    //     for (auto i = 0u; i < game.sxs; ++i)
    //     {
    //         uint8_t red = i;
    //         uint8_t green = j;
    //         uint8_t blue = (i * j);
    //         ofs << red << green << blue;
    //     }
    // }
    return 0;
}
