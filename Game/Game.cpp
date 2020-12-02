#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(FORWARD);
    }
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(BACKWARD);
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(STOPPED);
    }
    else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(RIGHT);
    }
    else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(LEFT);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setZ(-100);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setZ(0);
    }
    else if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        auto world = static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window));
        std::cout << world->getX() << ", " << world->getY() << std::endl;
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(STOP);
    }
}

int main()
{

    const unsigned int WINDOW_WIDTH = 640;
    const unsigned int WINDOW_HEIGHT = 480;

    // TODO: flip the map?

    std::vector<std::vector<int>> WORLD_MAP = {
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 1, 0, 1},
        {1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1},
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1},
        {1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1},
        {1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 3},
        {1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1, 0, 1},
        {1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1},
        {1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 3, 1},
        {1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3},
        {1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 2, 1, 0, 1},
        {1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
        {1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1},
        {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 1},
        {1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 3, 1},
        {1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1},
        {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1},
    };

    TexDict TEX_FNAMES = {
        {0, "../textures/wood.png"},
        {1, "../textures/redbrick.png"},
        {2, "../textures/redbrick.png"},
        {3, "../textures/redbrick.png"},
        {4, "../textures/arrow-left.png"},
        {5, "../textures/arrow-right.png"},
    };

    DisplayArray displayer(WINDOW_WIDTH, WINDOW_HEIGHT, keyCallback);

    RaycastWorld world(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_MAP, TEX_FNAMES);
    world.setDirection(0);
    world.addSprite(11.9, 1.5, 4);

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
