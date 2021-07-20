#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iostream>
#include <vector>

// TODO: remove as global variable
std::string image_directory;

/*
 * Saves an image from the window for the specified move
 * assumes image_directory doesn't include '/'
 * assumes image_directory/move directory already exists
 *
 * @param w RaycastWorld
 * @param move specified move and subdirectory to save image to
 */
void save_png_with_command(RaycastWorld *world, std::string move)
{
    static int count = 0;

    std::string image_path = image_directory + "/" + move + "/" + std::to_string(count) + ".png";
    std::cout << "Saving image \"" << image_path << "\"\n";
    world->save_png(image_path);
    count++;
}

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{

    auto world = static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window));

    if (image_directory.length() > 0 && (action == GLFW_PRESS || action == GLFW_REPEAT))
    {
        if (key == GLFW_KEY_UP)
        {
            save_png_with_command(world, "forward");
        }
        else if (key == GLFW_KEY_LEFT)
        {
            save_png_with_command(world, "left");
        }
        else if (key == GLFW_KEY_RIGHT)
        {
            save_png_with_command(world, "right");
        }
    }

    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    else if (key == GLFW_KEY_UP && action == GLFW_PRESS)
    {
        world->set_walk(Walk::FORWARD);
    }
    else if (key == GLFW_KEY_DOWN && action == GLFW_PRESS)
    {
        world->set_walk(Walk::BACKWARD);
    }
    else if ((key == GLFW_KEY_UP || key == GLFW_KEY_DOWN) && action == GLFW_RELEASE)
    {
        world->set_walk(Walk::STOP);
    }
    else if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS)
    {
        world->set_turn(Turn::RIGHT);
    }
    else if (key == GLFW_KEY_LEFT && action == GLFW_PRESS)
    {
        world->set_turn(Turn::LEFT);
    }
    else if ((key == GLFW_KEY_LEFT || key == GLFW_KEY_RIGHT) && action == GLFW_RELEASE)
    {
        world->set_turn(Turn::STOP);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_PRESS)
    {
        world->set_z(-100);
    }
    else if (key == GLFW_KEY_RIGHT_SHIFT && action == GLFW_RELEASE)
    {
        world->set_z(0);
    }
    else if (key == GLFW_KEY_P && action == GLFW_PRESS)
    {
        auto angle = world->get_direction() * 180.0 / 3.1415926;
        std::cout << world->get_x() << ", " << world->get_y() << " @ " << angle << std::endl;
    }
    else if (key == GLFW_KEY_M && action == GLFW_PRESS)
    {
        world->toggle_mini_map();
    }
    else if (key == GLFW_KEY_MINUS && action == GLFW_PRESS)
    {
        auto new_fov = world->get_fov();
        world->set_fov(new_fov - 5 * 3.1415926 / 180.0);
        std::cout << world->get_fov() * 180.0 / 3.1415926 << " degrees" << std::endl;
    }
    else if (key == GLFW_KEY_EQUAL && action == GLFW_PRESS)
    {
        auto new_fov = world->get_fov();
        world->set_fov(new_fov + 5 * 3.1415926 / 180.0);
        std::cout << world->get_fov() * 180.0 / 3.1415926 << " degrees" << std::endl;
    }
}

int main(int argc, char const *argv[])
{
    const unsigned int DEFAULT_WINDOW_WIDTH = 244;
    const unsigned int DEFAULT_WINDOW_HEIGHT = 244;
    const auto DEFAULT_WORLD_FILE = "../Mazes/maze01.txt";

    // Process program arguments (must be given in this order)
    std::string world_filepath = argc >= 2 ? argv[1] : DEFAULT_WORLD_FILE;
    image_directory = argc >= 3 ? argv[2] : "";
    usize width = argc >= 4 ? std::stoul(argv[3]) : DEFAULT_WINDOW_WIDTH;
    usize height = argc >= 5 ? std::stoul(argv[4]) : DEFAULT_WINDOW_HEIGHT;

    DisplayArray displayer(width, height, keyCallback);
    RaycastWorld world(width, height, world_filepath);

    glfwSetWindowUserPointer(displayer.window, &world);

    //double t = glfwGetTime();
    while (displayer.running())
    {
        displayer.pre();

        world.update_pose();
        displayer.render(world.get_buffer());

        displayer.post();

        //std::cout << 1 / (glfwGetTime() - t) << std::endl;
        //t = glfwGetTime();
    }

    return EXIT_SUCCESS;
}
