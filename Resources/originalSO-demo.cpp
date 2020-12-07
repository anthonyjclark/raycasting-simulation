#include <iostream>
using std::cerr;
using std::cout;

#include <fstream>
using std::ofstream;

#include "originalSO-compile.h"

struct Color
{
    byte r;
    byte g;
    byte b;
};

int main(/*int argc, char const *argv[]*/)
{
    Doom3D game;
    game.scr_resize(512, 512);
    game.draw();

    ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << game.sxs << ' ' << game.sys << '\n'
        << "255\n";
    // ofs.write(reinterpret_cast<char *>(game.scr), game.sxs * game.sys * sizeof(game.scr[0]));

    Color c;
    for (int i = 0; i < game.sxs * game.sys; i++)
    {
        c.r = (byte)((game.scr[i] & 0xFF0000) >> 16);
        c.g = (byte)((game.scr[i] & 0x00FF00) >> 8),
        c.b = (byte)((game.scr[i] & 0x0000FF));
        ofs.write(reinterpret_cast<char *>(&c), sizeof(c));
    }

    return EXIT_SUCCESS;
}
