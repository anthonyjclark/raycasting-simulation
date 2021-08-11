#include "DisplayArray.h"
#include "../RaycastWorld/RaycastWorld.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// rsync -arv images/ dgx01:/raid/clark/summer2021/datasets/handmade-full

// TODO: remove as global variable
std::string image_directory;
int image_number;

/*
 * Saves an image from the window for the specified move
 * assumes image_directory doesn't include '/'
 * assumes image_directory/move directory already exists
 *
 * @param w RaycastWorld
 * @param move specified move and subdirectory to save image to
 */
void save_png_with_command(RaycastWorld *world)
{
    // Set angle based on current turn and walk values (only works for classification)
    // TODO: make this work with regularization
    std::string angle;
    if (world->get_walk() == Walk::STOP && world->get_turn() == Turn::LEFT)
    {
        angle = "1p000";
    }
    else if (world->get_walk() == Walk::STOP && world->get_turn() == Turn::RIGHT)
    {
        angle = "-1p000";
    }
    else if (world->get_walk() == Walk::FORWARD && world->get_turn() == Turn::STOP)
    {
        angle = "0p000";
    }
    else
    {
        std::cout << "WARNING: cannot handle case when walking and turning at the same time.\n";
        return;
    }

    // From python: f"{i:>06}_{angle:.3f}".replace(".", "p") + ".png"
    std::ostringstream image_path;
    image_path << image_directory << (image_directory[image_directory.length() - 1] == '/' ? "" : "/")
               << std::setfill('0') << std::setw(6) << image_number
               << "_"
               << angle
               << ".png";
    std::cout << "Saving image \"" << image_path.str() << "\"\n";
    world->save_png(image_path.str());
    image_number++;
}

// void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
void keyCallback(GLFWwindow *window, int key, int, int action, int)
{
    auto world = static_cast<RaycastWorld *>(glfwGetWindowUserPointer(window));

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
    const auto DEFAULT_WORLD_FILE = "../Mazes/maze01.txt";
    const std::string DEFAULT_IMAGE_DIRECTORY = "";
    const int DEFAULT_IMAGE_NUMBER_START = 0;
    const unsigned int DEFAULT_WINDOW_WIDTH = 224;
    const unsigned int DEFAULT_WINDOW_HEIGHT = 224;

    // Process program arguments (must be given in this order)
    std::string world_filepath = argc >= 2 ? argv[1] : DEFAULT_WORLD_FILE;
    image_directory = argc >= 3 ? argv[2] : DEFAULT_IMAGE_DIRECTORY;
    image_number = argc >= 4 ? std::stol(argv[3]) : DEFAULT_IMAGE_NUMBER_START;
    usize width = argc >= 5 ? std::stoul(argv[4]) : DEFAULT_WINDOW_WIDTH;
    usize height = argc >= 6 ? std::stoul(argv[5]) : DEFAULT_WINDOW_HEIGHT;

    DisplayArray displayer(width, height, keyCallback);
    RaycastWorld world(width, height, world_filepath);

    glfwSetWindowUserPointer(displayer.window, &world);

    //double t = glfwGetTime();
    while (displayer.running())
    {
        displayer.pre();

        world.update_pose();

        // Save an image if the world was updated
        if (image_directory.length() > 0 && world.in_motion())
        {
            save_png_with_command(&world);
        }

        displayer.render(world.get_buffer());

        displayer.post();

        //std::cout << 1 / (glfwGetTime() - t) << std::endl;
        //t = glfwGetTime();
    }

    return EXIT_SUCCESS;
}
