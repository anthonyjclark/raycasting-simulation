#include "RaycastWorld.h"

int main(/*int argc, char const *argv[]*/)
{
    const unsigned int IMAGE_WIDTH = 640;
    const unsigned int IMAGE_HEIGHT = 480;

    RaycastWorld world(IMAGE_WIDTH, IMAGE_HEIGHT, "../Mazes/maze.txt");

    world.toggleMiniMap();
    world.setPosition(1.5, 1.5);

    world.setDirection(0);
    world.savePNG("frame0deg.png");

    world.setDirection(1.57);
    world.savePNG("frame90deg.png");

    return EXIT_SUCCESS;
}
