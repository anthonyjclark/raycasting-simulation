#include "RaycastWorld.h"

int main(/*int argc, char const *argv[]*/)
{
    const unsigned int IMAGE_WIDTH = 640;
    const unsigned int IMAGE_HEIGHT = 480;

    RaycastWorld world(IMAGE_WIDTH, IMAGE_HEIGHT, "../Mazes/maze01.txt");

    world.toggle_mini_map();
    world.set_position(1.5, 1.5);

    world.set_direction(0);
    world.save_png("frame0deg.png");

    world.set_direction(1.57);
    world.save_png("frame90deg.png");

    return EXIT_SUCCESS;
}
