#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

// TODO: remove as global variable
std::string imageDirectory;

/*
 * Saves an image from the window for the specified move
 * assumes imageDirectory doesn't include '/'
 * assumes imageDirectory/move directory already exists
 *
 * @param w RaycastWorld
 * @param move specified move and subdirectory to save image to
 */
void saveCommandPNG(RaycastWorld *world, std::string move)
{
    static int count = 0;
    std::string imagePath = imageDirectory + "/" + move + "/" + std::to_string(count) + ".png";
    std::cout << "Saving image \"" << imagePath << "\"\n";
    world->savePNG(imagePath);
    count++;
}

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    auto world = static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window));

    if (imageDirectory.length() > 0 && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        if (key == GLFW_KEY_UP)
        {
            saveCommandPNG(world, "forward");
        }
        else if (key == GLFW_KEY_LEFT)
        {
            saveCommandPNG(world, "left");
        }
        else if (key == GLFW_KEY_RIGHT)
        {
            saveCommandPNG(world, "right");
        }
    }

    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        world->setWalk(FORWARD);
    }
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        world->setWalk(BACKWARD);
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {
        world->setWalk(STOPPED);
    }
    else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
    {
        world->setTurn(RIGHT);
    }
    else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        world->setTurn(LEFT);
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_RELEASE)
    {
        world->setTurn(STOP);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_PRESS)
    {
        world->setZ(-100);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_RELEASE)
    {
        world->setZ(0);
    }
    else if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        std::cout << world->getX() << ", " << world->getY() << std::endl;
    }
    else if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        world->toggleMiniMap();
    }
}

int main(int argc, char const *argv[])
{
    const unsigned int DEFAULT_WINDOW_WIDTH = 320;
    const unsigned int DEFAULT_WINDOW_HEIGHT = 240;
    const auto DEFAULT_WORLD_FILE = "../Worlds/maze.txt";

    // Process program arguments (must be given in this order)
    std::string worldFilepath = argc >= 2 ? argv[1] : DEFAULT_WORLD_FILE;
    imageDirectory = argc >= 3 ? argv[2] : "";
    usize width = argc >= 4 ? std::stoul(argv[3]) : DEFAULT_WINDOW_WIDTH;
    usize height = argc >= 5 ? std::stoul(argv[4]) : DEFAULT_WINDOW_HEIGHT;

    DisplayArray displayer(width, height, keyCallback);
    RaycastWorld world(width, height, worldFilepath);

    glfwSetWindowUserPointer(displayer.window, &world);

    //double t = glfwGetTime();
    while (displayer.running())
    {
        displayer.pre();

        world.updatePose();
        displayer.render(world.getBuffer());

        displayer.post();

        //std::cout << 1 / (glfwGetTime() - t) << std::endl;
        //t = glfwGetTime();
    }

    return EXIT_SUCCESS;
}
