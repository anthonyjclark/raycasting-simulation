#include "RaycastWorld.h"

// rm frame.ppm && make && ./world && open frame.ppm

int main(/*int argc, char const *argv[]*/)
{

    const unsigned int WINDOW_WIDTH = 640;
    const unsigned int WINDOW_HEIGHT = 480;

    const std::vector<std::vector<int>> WORLD_MAP = {
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2},
        {2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2},
        {2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2},
        {2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 0, 2},
        {2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2},
        {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2},
    };

    RaycastWorld world(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_MAP);

    std::ofstream ofs("frame.ppm", std::ios::out | std::ios::binary);
    ofs << "P6\n"
        << world.getWidth() << ' ' << world.getHeight() << '\n'
        << "255\n";

    ofs.write(reinterpret_cast<char *>(world.getBuffer()), sizeof(uint8_t) * world.getWidth() * world.getHeight() * 3);

    return 0;
}
