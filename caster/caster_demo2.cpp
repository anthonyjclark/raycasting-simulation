#include "caster2.h"

int main(/*int argc, char const *argv[]*/)
{
    Caster caster(640, 480);

    std::ofstream ofs("first.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << caster.width() << ' ' << caster.height() << '\n'
        << "255\n";

    ofs.write(reinterpret_cast<char *>(caster.getBuffer()), sizeof(uint8_t) * caster.width() * caster.height() * 3);

    return 0;
}
