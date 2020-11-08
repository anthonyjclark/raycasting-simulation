#include "caster2.h"

int main(/*int argc, char const *argv[]*/)
{
    Caster caster;

    std::ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << screenWidth << ' ' << screenHeight << '\n'
        << "255\n";

    ofs.write(reinterpret_cast<char *>(caster.buffer.data()), sizeof(uint8_t) * screenWidth * screenHeight * 3);
    // for (int y = 0; y < screenHeight; y++)
    // {
    //     for (int x = 0; x < screenWidth; x++)
    //     {
    //         // auto bc = caster.buffer[y][x];
    //         auto bc = caster.buffer[x + y * screenWidth];
    //         ColorRGB8bit c;
    //         c.r = (bc & 0xFF0000) >> 16;
    //         c.g = (bc & 0xFF00) >> 8;
    //         c.b = (bc & 0xFF);
    //         ofs.write(reinterpret_cast<char *>(&c), sizeof(c));
    //     }
    // }

    return 0;
}
