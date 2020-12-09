#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

int count = 0;

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {   
        std::cout << "Total moves: " + std::to_string(::count) + "\n";
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {   
        std::string imagePath = "../Images/forward/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        // std::cout << "Forward\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->savePNG(pngPath);
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(FORWARD);
        ::count++;
    }
    else if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {   
        std::string imagePath = "../Images/backward/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        // std::cout << "Backward\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->savePNG(pngPath);
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(BACKWARD);
        ::count++;
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {   
        std::string imagePath = "../Images/stop/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        // std::cout << "Stop\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->savePNG(pngPath);
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(STOPPED);
        ::count++;
    }
    else if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        std::string imagePath = "../Images/right/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        // std::cout << "Right\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->savePNG(pngPath);
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(RIGHT);
        ::count++;
    }
    else if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        std::string imagePath = "../Images/left/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->savePNG(pngPath);
        // std::cout << "Left\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(LEFT);
        ::count++;
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
    else if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->toggleMiniMap();
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(STOP);
    }
}

int main(int argc, char const *argv[])
{
    // const unsigned int WINDOW_WIDTH = 640;
    // const unsigned int WINDOW_HEIGHT = 480;
    const unsigned int WINDOW_WIDTH = 320;
    const unsigned int WINDOW_HEIGHT = 240;

    std::string mapFile = argc >= 2 ? argv[1] : "../Mazes/maze.txt";
    usize width = argc >= 4 ? std::stoul(argv[2]) : WINDOW_WIDTH;
    usize height = argc >= 4 ? std::stoul(argv[2]) : WINDOW_HEIGHT;

    DisplayArray displayer(width, height, keyCallback);
    RaycastWorld world(width, height, mapFile);
    world.setDirection(0);

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
