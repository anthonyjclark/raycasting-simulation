#include "caster2.h"

int main(/*int argc, char const *argv[]*/)
{
    Caster caster;

    std::ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << screenWidth << ' ' << screenHeight << '\n'
        << "255\n";

    ofs.write(reinterpret_cast<char *>(caster.buffer.data()), sizeof(uint8_t) * screenWidth * screenHeight * 3);

    return 0;
}
