#include "caster2.h"

// rm frame.ppm && make && ./caster && open frame.ppm

int main(/*int argc, char const *argv[]*/)
{
    Caster caster(640, 480);
    caster.render(22.0, 11.5);

    std::ofstream ofs("frame.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << caster.width() << ' ' << caster.height() << '\n'
        << "255\n";

    ofs.write(reinterpret_cast<char *>(caster.getBuffer()), sizeof(uint8_t) * caster.width() * caster.height() * 3);

    return 0;
}
