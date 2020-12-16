#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>
#include <random>

std::string imgDir;

/*
 * Saves an image from the window for the specified move
 * assumes imgDir doesn't include '/'
 * assumes imgDir/move directory already exists
 * 
 * @param w RaycastWorld
 * @param move specified move and subdirectory to save image to
 */
void saveImg(RaycastWorld *w, std::string move)
{
    static int count = 0;
    if (move == "q")
    {
        std::cout << "Total moves " + std::to_string(count) + "\n";
    }
    else if (imgDir != "none")
    {
        std::string imgPath = imgDir + "/" + move + "/" + std::to_string(count) + ".png";
        w->savePNG(imgPath);
        count++;
    }
}

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    // std::cout << key << std::endl;
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        saveImg(static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window)), "q");
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        saveImg(static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window)), "forward");
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(FORWARD);
    }
    else if (key == GLFW_KEY_DOWN && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        // saveImg(static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window)), "backward");
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(BACKWARD);
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(STOPPED);
    }
    else if (key == GLFW_KEY_RIGHT && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        saveImg(static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window)), "right");
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(RIGHT);
    }
    else if (key == GLFW_KEY_LEFT && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        saveImg(static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window)), "left");
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

    // static int count = std::stoul(argv[1]);
    std::string mapFile = argc >= 2 ? argv[1] : "../Worlds/maze.txt";
    imgDir = argc >= 3 ? argv[2] : "none";

    usize width = argc >= 4 ? std::stoul(argv[3]) : WINDOW_WIDTH;
    usize height = argc >= 5 ? std::stoul(argv[4]) : WINDOW_HEIGHT;

    DisplayArray displayer(width, height, keyCallback);
    RaycastWorld world(width, height, mapFile);

    // hard coded ranges
    double lower_dir = 3.14 / 4;
    double upper_dir = 3.14 * 3 / 4;
    double lower_posx = 17;
    double upper_posx = 18;
    double lower_posy = 9;
    double upper_posy = 10;
    // double lower_posx = 3;
    // double upper_posx = 4;
    // double lower_posy = 15;
    // double upper_posy = 16;

    std::random_device rd;
    std::default_random_engine eng(rd());

    std::uniform_real_distribution<double> distr_dir(lower_dir, upper_dir);
    std::uniform_real_distribution<double> distr_x(lower_posx, upper_posx);
    std::uniform_real_distribution<double> distr_y(lower_posy, upper_posy);

    double rand_dir = distr_dir(eng);
    double rand_x = distr_x(eng);
    double rand_y = distr_y(eng);

    // setting direction to random value within specified range
    world.setDirection(rand_dir);

    // setting random position on path within specified range
    world.setPosition(rand_x, rand_y);

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
