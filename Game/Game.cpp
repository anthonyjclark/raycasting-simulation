#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
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
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(STOP);
    }
}

int main()
{

    const unsigned int WINDOW_WIDTH = 640;
    const unsigned int WINDOW_HEIGHT = 480;

    std::vector<std::vector<int>> WORLD_MAP = {
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

    DisplayArray displayer(WINDOW_WIDTH, WINDOW_HEIGHT, keyCallback);
    RaycastWorld world(WINDOW_WIDTH, WINDOW_HEIGHT, WORLD_MAP);
    glfwSetWindowUserPointer(displayer.window, &world);

    //double t = glfwGetTime();
    while (displayer.running())
    {
        displayer.pre();
        world.updatePose();
        world.renderView();
        displayer.render(world.getBuffer());
        displayer.post();

        //std::cout << 1 / (glfwGetTime() - t) << std::endl;
        //t = glfwGetTime();
    }

    return EXIT_SUCCESS;
}
