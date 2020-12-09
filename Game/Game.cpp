#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

#include <stdio.h>
#include <stdlib.h>

// #define STB_IMAGE_IMPLEMENTATION
// #include "../stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image/stb_image_write.h"

using namespace std;

int count = 0;

// https://lencerf.github.io/post/2019-09-21-save-the-opengl-rendering-to-image-file/
void saveImage(char* filepath, GLFWwindow* w) 
{
    int width, height;
    glfwGetFramebufferSize(w, &width, &height);
    GLsizei nrChannels = 3;
    GLsizei stride = nrChannels * width;
    stride += (stride % 4) ? (4 - stride % 4) : 0;
    GLsizei bufferSize = stride * height;
    std::vector<char> buffer(bufferSize);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadBuffer(GL_FRONT);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, buffer.data());
    stbi_flip_vertically_on_write(true);
    stbi_write_png(filepath, width, height, nrChannels, buffer.data(), stride);
}

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {   
        std::string imagePath = "../Images/forward/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        saveImage(pngPath, window);
        std::cout << "Forward\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(FORWARD);
        ::count++;
    }
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {   
        std::string imagePath = "../Images/backward/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        saveImage(pngPath,  window);
        std::cout << "Backward\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(BACKWARD);
        ::count++;
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {   
        std::string imagePath = "../Images/stop/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        saveImage(pngPath,  window);
        std::cout << "Stop\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setWalk(STOPPED);
        ::count++;
    }
    else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
    {
        std::string imagePath = "../Images/right/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        saveImage(pngPath,  window);
        std::cout << "Right\n";
        static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window))->setTurn(RIGHT);
        ::count++;
    }
    else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        std::string imagePath = "../Images/left/" + std::to_string(::count) + ".png";
        char* pngPath = &imagePath[0];
        saveImage(pngPath,  window);
        std::cout << "Left\n";
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
    const unsigned int WINDOW_WIDTH = 640;
    const unsigned int WINDOW_HEIGHT = 480;

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
