// TODO:
// - take into account dt
// - use gpu? parallelize rays?
// - set ceiling and floor textures
// - license
// - generalize floor, ceiling, and checkerboard
// - allow solid colors (use something other than int for each cell?)
// - allow to set turn and walk speeds

#if !defined(_RAYCAST_WORLD_H_)
#define _RAYCAST_WORLD_H_

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include <lodepng.h>

// Parameters for scaling and moving the sprites
#define uDiv 1
#define vDiv 1
#define vMove 0.0

using usize = uint32_t;
using uintrgb = uint8_t;

using World = std::vector<std::vector<int>>;
using Tex = std::vector<uintrgb>;
using TexMap = std::unordered_map<usize, Tex>;

// Texture dimensions must be power of two
const usize tex_width = 1024;
const usize tex_height = 1024;

struct Sprite
{
    double x;
    double y;
    usize texture;
    usize order;
    double distance2;
};

enum class Hit
{
    X,
    Y
};

enum class Turn
{
    LEFT = -1,
    STOP = 0,
    RIGHT = 1,
};

enum class Walk
{
    BACKWARD = -1,
    STOP = 0,
    FORWARD = 1,
};

double deg2rad(double deg)
{
    return deg * 3.1415926 / 180.0;
}

void read_png(std::vector<uintrgb> &out, const std::string &filename)
{
    unsigned width, height;
    unsigned error = lodepng::decode(out, width, height, filename, LCT_RGB, 8);

    if (error)
    {
        std::cerr << "Error reading PNG " << error << ": " << lodepng_error_text(error) << std::endl;
    }
}

class RaycastWorld
{
private:
    // World map
    usize map_width;
    usize map_height;
    World world_map;
    bool show_mini_map;

    // Render buffer
    usize screen_width;
    usize screen_height;
    std::vector<uintrgb> buffer;

    // Write a single color pixel to the buffer
    inline void write_hex_color(int x, int y, usize c, bool darken = false)
    {
        c = darken ? (c >> 1) & 0x7F : c;
        buffer[(x * 3 + 0) + y * screen_width * 3] = (c & 0xFF0000) >> 16;
        buffer[(x * 3 + 1) + y * screen_width * 3] = (c & 0xFF00) >> 8;
        buffer[(x * 3 + 2) + y * screen_width * 3] = (c & 0xFF);
    }

    inline void write_rgb_color(int x, int y, uintrgb *c, bool darken = false)
    {
        buffer[(x * 3 + 0) + y * screen_width * 3] = darken ? (c[0] >> 1) & 0x7F : c[0];
        buffer[(x * 3 + 1) + y * screen_width * 3] = darken ? (c[1] >> 1) & 0x7F : c[1];
        buffer[(x * 3 + 2) + y * screen_width * 3] = darken ? (c[2] >> 1) & 0x7F : c[2];
    }

    // Texture buffers
    TexMap textures;

    // 1D Zbuffer for sprites
    std::vector<double> z_buffer;

    // Arrays used to sort the sprites
    std::vector<Sprite> sprites;

    void render_floor_and_ceiling();
    void render_walls();
    void render_sprites();
    void render_mini_map();

    // View and camera pose
    double pos_x, pos_y, pos_z;
    double pitch;
    double dir_x, dir_y;
    double plane_x, plane_y;
    double fov;

    // Initial view
    double initial_x, initial_y, initial_angle, initial_fov;

    // Goal view
    double goal_x, goal_y;

    // Motion
    Turn turn_direction;
    double turn_speed;

    Walk walk_direction;
    double walk_speed;

    bool need_to_render;

public:
    RaycastWorld() = delete;
    RaycastWorld(usize width, usize height, std::string maze_file_path);

    void update_pose();
    void render_view();

    void set_turn(Turn turn) { turn_direction = turn; }
    void set_walk(Walk walk) { walk_direction = walk; }

    Turn get_turn() { return turn_direction; }
    Walk get_walk() { return walk_direction; }

    bool in_motion() { return turn_direction != Turn::STOP || walk_direction != Walk::STOP; }

    auto get_screen_width() { return screen_width; }
    auto get_screen_height() { return screen_height; }

    auto get_x() { return pos_x; }
    auto get_y() { return pos_y; }
    auto get_z() { return pos_z; }

    auto get_dir_x() { return dir_x; }
    auto get_dir_y() { return dir_y; }
    auto get_direction() { return atan2(dir_y, dir_x); }
    auto get_fov() { return fov; }

    auto get_turn_speed() { return turn_speed; }
    auto get_walk_speed() { return walk_speed; }

    void set_x(double x)
    {
        pos_x = x;
        need_to_render = true;
    }

    void set_y(double y)
    {
        pos_y = y;
        need_to_render = true;
    }

    void set_z(double z)
    {
        pos_z = z;
        need_to_render = true;
    }

    void set_position(double x, double y, double z = 0.0)
    {
        pos_x = x;
        pos_y = y;
        pos_z = z;
        need_to_render = true;
    }

    // Useful for Python bindings
    void set_position_xy(double x, double y) { set_position(x, y); }

    void set_direction(double radians)
    {
        dir_x = cos(radians);
        dir_y = sin(radians);

        // 2 * atan(L/1.0) = fov
        // L = tan(fov/2)
        double plane_length = tan(fov / 2);
        plane_x = plane_length * sin(radians);
        plane_y = -plane_length * cos(radians);

        need_to_render = true;
    }

    void set_fov(double radians)
    {
        fov = radians;
        initial_fov = radians; // TODO: this should be removed when fov is a paramter
        set_direction(get_direction());
        need_to_render = true;
    }

    void reset()
    {
        set_position(initial_x, initial_y);
        set_fov(initial_fov);
        set_direction(initial_angle);
        turn_direction = Turn::STOP;
        walk_direction = Walk::STOP;
    }

    bool at_goal()
    {
        return fabs(pos_x - goal_x) < 1 && fabs(pos_y - goal_y) < 1;
    }

    void toggle_mini_map()
    {
        show_mini_map = !show_mini_map;
        need_to_render = true;
    }

    auto get_buffer()
    {
        render_view();
        return buffer.data();
    }

    void save_png(std::string filename)
    {
        render_view();
        auto error = lodepng::encode(filename, buffer, screen_width, screen_height, LCT_RGB, 8);
        if (error)
        {
            std::cerr << "Error writing PNG " << error << ": " << lodepng_error_text(error) << std::endl;
        }
    }

    void add_sprite(double x, double y, usize tex_id)
    {
        Sprite sp{x, y, tex_id, 1, 0.0};
        sprites.push_back(sp);
    }
};

RaycastWorld::RaycastWorld(usize width, usize height, std::string maze_file_path)
    : screen_width(width), screen_height(height), buffer(width * height * 3), z_buffer(width)
{
    std::ifstream maze_file(maze_file_path);
    if (!maze_file.is_open())
    {
        std::cerr << "Error opening maze file." << std::endl;
        return;
    }

    usize num_textures = 0;
    maze_file >> num_textures;
    std::cout << "Number of texture: " << num_textures << std::endl;

    // Loop through given texture files
    for (usize tex_id = 0; tex_id < num_textures; ++tex_id)
    {
        std::string tex_filename;
        maze_file >> tex_filename;
        std::cout << "Texture filename: " << tex_filename << std::endl;

        // Tex texture(tex_width * tex_height * 3);
        Tex texture; // TODO: does this need to be pre-sized?
        read_png(texture, tex_filename);
        textures[tex_id] = texture;
    }

    maze_file >> map_width >> map_height;
    std::cout << "Map is " << map_width << " by " << map_height << std::endl;

    for (usize row = 0; row < map_height; ++row)
    {
        std::vector<int> mapRow;
        for (usize col = 0; col < map_width; ++col)
        {
            int cell_value;
            maze_file >> cell_value;
            mapRow.push_back(cell_value);
        }
        world_map.emplace_back(mapRow);
    }

    // Map needs to be reversed so that the bottom left is 0,0
    std::reverse(world_map.begin(), world_map.end());

    show_mini_map = false;

    // TODO: add sprites to maze file?
    // Add the sprites
    // double x, y;
    // usize tex_id;
    // while (maze_file >> x >> y >> tex_id)
    // {
    //     add_sprite(x, y, tex_id);
    //     break;
    // }

    // Location on the map
    std::string direction;
    maze_file >> pos_x >> pos_y >> direction;

    // Position into middle of cell
    pos_x += 0.5;
    pos_y += 0.5;

    initial_x = pos_x;
    initial_y = pos_y;

    // Vertical camera strafing up/down, for jumping/crouching.
    // 0 means standard height.
    // Expressed in screen pixels a wall at distance 1 shifts
    pos_z = 0;

    // Looking up/down, expressed in screen pixels the horizon shifts
    pitch = 0;

    // Direction vector
    dir_x = -1.0;
    dir_y = 0.0;

    // The 2d raycaster version of camera plane
    plane_x = 0.0;
    plane_y = 0.66;

    // Motion and rendering
    turn_direction = Turn::STOP;
    turn_speed = 2.5 * (3.1415926 / 180);
    walk_direction = Walk::STOP;
    walk_speed = 0.05;

    // Set direction
    if (direction == "Dir.NORTH")
    {
        initial_angle = deg2rad(90);
    }
    else if (direction == "Dir.EAST")
    {
        initial_angle = deg2rad(0);
    }
    else if (direction == "Dir.SOUTH")
    {
        initial_angle = deg2rad(270);
    }
    else
    {
        initial_angle = deg2rad(180);
    }

    // TODO: make FOV a parameter
    fov = 75.0 * 3.1415926 / 180.0;
    initial_fov = fov;
    set_direction(initial_angle);

    std::cout << "Initial pose: "
              << initial_x << ", " << initial_y << " "
              << initial_angle << " (" << direction << ")" << std::endl;

    // Loop through maze file to get coordinates of the goal
    std::string goal_dir;
    while (maze_file >> goal_x >> goal_y >> goal_dir)
        ;
    std::cout << "Goal position: "
              << goal_x << ", " << goal_y << std::endl;

    need_to_render = true;
    render_view();
}

void RaycastWorld::update_pose()
{
    if (turn_direction != Turn::STOP)
    {
        double rot_speed = static_cast<int>(turn_direction) * turn_speed;
        double old_dir_x = dir_x;
        dir_x = old_dir_x * cos(-rot_speed) - dir_y * sin(-rot_speed);
        dir_y = old_dir_x * sin(-rot_speed) + dir_y * cos(-rot_speed);

        double old_plane_x = plane_x;
        plane_x = old_plane_x * cos(-rot_speed) - plane_y * sin(-rot_speed);
        plane_y = old_plane_x * sin(-rot_speed) + plane_y * cos(-rot_speed);

        need_to_render = true;
    }

    if (walk_direction != Walk::STOP)
    {
        double step = static_cast<int>(walk_direction) * walk_speed;
        double new_x = pos_x + dir_x * step;
        double new_y = pos_y + dir_y * step;

        // TODO: x vs y?

        if (world_map[int(pos_y)][int(new_x)] == 0)
        {
            pos_x = new_x;
            need_to_render = true;
        }

        if (world_map[int(new_y)][int(pos_x)] == 0)
        {
            pos_y = new_y;
            need_to_render = true;
        }
    }
}

void RaycastWorld::render_view()
{
    if (need_to_render)
    {
        render_floor_and_ceiling();
        render_walls();
        render_sprites();

        if (show_mini_map)
        {
            render_mini_map();
        }
        need_to_render = false;
    }
}

void RaycastWorld::render_floor_and_ceiling()
{
    for (auto y = 0u; y < screen_height; ++y)
    {
        // Whether this section is floor or ceiling
        bool is_floor = y > screen_height / 2 + pitch;

        // ray_dir for leftmost ray (x = 0) and rightmost ray (x = screen_width)
        float ray_dir_x0 = dir_x - plane_x;
        float ray_dir_y0 = dir_y - plane_y;
        float ray_dir_x1 = dir_x + plane_x;
        float ray_dir_y1 = dir_y + plane_y;

        // Current y position compared to the center of the screen (the horizon)
        int p = is_floor ? (y - screen_height / 2 - pitch) : (screen_height / 2 - y + pitch);

        // Vertical position of the camera.
        // NOTE: with 0.5, it's exactly in the center between floor and ceiling,
        // matching also how the walls are being raycasted. For different values
        // than 0.5, a separate loop must be done for ceiling and floor since
        // they're no longer symmetrical.
        float cam_z = is_floor ? (0.5 * screen_height + pos_z) : (0.5 * screen_height - pos_z);

        // Horizontal distance from the camera to the floor for the current row.
        // 0.5 is the z position exactly in the middle between floor and ceiling.
        // NOTE: this is affine texture mapping, which is not perspective correct
        // except for perfectly horizontal and vertical surfaces like the floor.
        // NOTE: this formula is explained as follows: The camera ray goes through
        // the following two points: the camera itself, which is at a certain
        // height (pos_z), and a point in front of the camera (through an imagined
        // vertical plane containing the screen pixels) with horizontal distance
        // 1 from the camera, and vertical position p lower than pos_z (pos_z - p). When going
        // through that point, the line has vertically traveled by p units and
        // horizontally by 1 unit. To hit the floor, it instead needs to travel by
        // pos_z units. It will travel the same ratio horizontally. The ratio was
        // 1 / p for going through the camera plane, so to go pos_z times farther
        // to reach the floor, we get that the total horizontal distance is pos_z / p.
        float row_distance = cam_z / p;

        // calculate the real world step vector we have to add for each x (parallel to camera plane)
        // adding step by step avoids multiplications with a weight in the inner loop
        float floor_step_x = row_distance * (ray_dir_x1 - ray_dir_x0) / screen_width;
        float floor_step_y = row_distance * (ray_dir_y1 - ray_dir_y0) / screen_width;

        // real world coordinates of the leftmost column. This will be updated as we step to the right.
        float floor_x = pos_x + row_distance * ray_dir_x0;
        float floor_y = pos_y + row_distance * ray_dir_y0;

        for (auto x = 0u; x < screen_width; ++x)
        {
            // The cell coord is simply got from the integer parts of floor_x and floor_y
            int cell_x = floor_x;
            int cell_y = floor_y;

            // Get the texture coordinate from the fractional part
            int tex_x = (int)(tex_width * (floor_x - cell_x)) & (tex_width - 1);
            int tex_y = (int)(tex_height * (floor_y - cell_y)) & (tex_height - 1);

            floor_x += floor_step_x;
            floor_y += floor_step_y;

            // Choose texture and draw the pixel
            // int checkerBoardPattern = (int(cell_x + cell_y)) & 1;
            // int floorTexture = checkerBoardPattern == 0 ? 1 : 2;
            // int ceilingTexture = 0;

            // // Floor or ceiling (and make it a bit darker)
            // if (is_floor)
            // {
            //     auto color = checkerBoardPattern == 0 ? 0x666666 : 0xEEEEEE;
            //     write_hex_color(x, y, (color >> 1) & 0x7F7F7F);
            // }
            // else
            // {
            //     int tex_id = ceilingTexture;
            //     write_rgb_color(x, y, &textures[tex_id][(tex_x + tex_width * tex_y) * 3]);
            // }
            int tex_id = is_floor ? 0 : 1;
            write_rgb_color(x, y, &textures[tex_id][(tex_x + tex_width * tex_y) * 3]);
        }
    }
}

void RaycastWorld::render_walls()
{
    for (auto x = 0u; x < screen_width; x++)
    {
        // x-coordinate in camera space
        double camera_x = 2 * x / double(screen_width) - 1;

        // Calculate ray position and direction
        double ray_dir_x = dir_x + plane_x * camera_x;
        double ray_dir_y = dir_y + plane_y * camera_x;

        // Which box of the map we're in
        int map_x = pos_x;
        int map_y = pos_y;

        // Length of ray from one x or y-side to next x or y-side
        double delta_dist_x = std::abs(1 / ray_dir_x);
        double delta_dist_y = std::abs(1 / ray_dir_y);
        double perp_wall_dist;

        // What direction to step in x or y-direction (either +1 or -1)
        int step_x = ray_dir_x < 0 ? -1 : 1;
        int step_y = ray_dir_y < 0 ? -1 : 1;

        // Length of ray from current position to next x or y-side
        double side_dist_x = ray_dir_x < 0 ? (pos_x - map_x) * delta_dist_x : (map_x + 1.0 - pos_x) * delta_dist_x;
        double side_dist_y = ray_dir_y < 0 ? (pos_y - map_y) * delta_dist_y : (map_y + 1.0 - pos_y) * delta_dist_y;

        // Was a NS or a EW wall hit?
        Hit side;

        // Perform DDA
        bool hit = false;
        while (hit == false)
        {
            // Jump to next map square, OR in x-direction, OR in y-direction
            if (side_dist_x < side_dist_y)
            {
                side_dist_x += delta_dist_x;
                map_x += step_x;
                side = Hit::X;
            }
            else
            {
                side_dist_y += delta_dist_y;
                map_y += step_y;
                side = Hit::Y;
            }

            // Check if ray has hit a wall
            hit = world_map[map_y][map_x] > 0;
        }

        // Calculate distance of perpendicular ray (Euclidean distance will give fisheye effect!)
        if (side == Hit::X)
        {
            perp_wall_dist = (map_x - pos_x + (1 - step_x) / 2) / ray_dir_x;
        }
        else
        {
            perp_wall_dist = (map_y - pos_y + (1 - step_y) / 2) / ray_dir_y;
        }

        // Calculate height of line to draw on screen
        int line_height = screen_height / perp_wall_dist;

        // Calculate lowest and highest pixel to fill in current stripe
        int draw_start = -line_height / 2 + screen_height / 2 + pitch + (pos_z / perp_wall_dist);
        draw_start = std::max(draw_start, 0);

        int draw_end = line_height / 2 + screen_height / 2 + pitch + (pos_z / perp_wall_dist);
        draw_end = std::min(draw_end, static_cast<int>(screen_height) - 1);

        // Texturing calculations (1 subtracted from it so that texture 0 can be used!)
        int tex_id = world_map[map_y][map_x];

        // Calculate value of wall_x (where exactly the wall was hit)
        double wall_x = side == Hit::X ? pos_y + perp_wall_dist * ray_dir_y : pos_x + perp_wall_dist * ray_dir_x;
        wall_x -= floor(wall_x);

        // x-coordinate on the texture
        int tex_x = int(wall_x * double(tex_width));
        if ((side == Hit::X && ray_dir_x > 0) || (side == Hit::Y && ray_dir_y < 0))
        {
            tex_x = tex_width - tex_x - 1;
        }

        // TODO: an integer-only bresenham or DDA like algorithm could make the texture coordinate stepping faster

        // How much to increase the texture coordinate per screen pixel
        double step = 1.0 * tex_height / line_height;

        // Starting texture coordinate
        double tex_pos = (draw_start - pitch - (pos_z / perp_wall_dist) - screen_height / 2 + line_height / 2) * step;

        for (int y = draw_start; y < draw_end; y++)
        {
            // Cast the texture coordinate to integer, and mask with (tex_height - 1) in case of overflow
            int tex_y = (int)tex_pos & (tex_height - 1);
            tex_pos += step;
            write_rgb_color(x, y, &textures[tex_id][(tex_x + tex_width * tex_y) * 3], side == Hit::Y);
        }

        // Set zbuffer as distance to wall for sprite casting
        z_buffer[x] = perp_wall_dist;
    }
}

void RaycastWorld::render_sprites()
{
    // Compute distances
    for (auto &sprite : sprites)
    {
        sprite.distance2 = (pos_x - sprite.x) * (pos_x - sprite.x) + (pos_y - sprite.y) * (pos_y - sprite.y);
    }

    // Sort by distance
    std::sort(sprites.begin(), sprites.end(), [](auto a, auto b)
              { return a.distance2 > b.distance2; });

    // Project and draw sprites
    // for (int i = 0; i < numSprites; i++)
    for (const auto &sprite : sprites)
    {
        // Translate sprite position to relative to camera
        double sprite_x = sprite.x - pos_x;
        double sprite_y = sprite.y - pos_y;

        // Transform sprite with the inverse camera matrix
        // [ plane_x   dir_x ] -1                                  [ dir_y      -dir_x ]
        // [               ]    =  1/(plane_x*dir_y-dir_x*plane_y) * [                 ]
        // [ plane_y   dir_y ]                                     [ -plane_y  plane_x ]

        // Required for correct matrix multiplication
        double inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y);

        double transform_x = inv_det * (dir_y * sprite_x - dir_x * sprite_y);
        // This is actually the depth inside the screen, that what Z is in 3D, the distance of
        // sprite to player, matching sqrt(spriteDistance[i])
        double transform_y = inv_det * (-plane_y * sprite_x + plane_x * sprite_y);

        int sprite_screen_x = (screen_width / 2) * (1 + transform_x / transform_y);

        int v_move_screen = int(vMove / transform_y) + pitch + pos_z / transform_y;

        // Calculate height of the sprite on screen
        // Using "transform_y" instead of the real distance prevents fisheye
        int sprite_height = abs(int(screen_height / (transform_y))) / vDiv;

        // Calculate lowest and highest pixel to fill in current stripe
        int draw_start_y = -sprite_height / 2 + screen_height / 2 + v_move_screen;
        draw_start_y = std::max(draw_start_y, 0);

        int draw_end_y = sprite_height / 2 + screen_height / 2 + v_move_screen;
        draw_end_y = std::min(draw_end_y, static_cast<int>(screen_height) - 1);

        // Calculate width of the sprite
        int sprite_width = abs(int(screen_height / (transform_y))) / uDiv;

        usize draw_start_x = std::max(-sprite_width / 2 + sprite_screen_x, 0);
        usize draw_end_x = std::max(std::min(sprite_width / 2 + sprite_screen_x, static_cast<int>(screen_width) - 1), 0);

        // Loop through every vertical stripe of the sprite on screen
        for (usize stripe = draw_start_x; stripe < draw_end_x; stripe++)
        {
            int tex_x = int(256 * (stripe - (-sprite_width / 2 + sprite_screen_x)) * tex_width / sprite_width) / 256;

            //the conditions in the if are:
            //1) it's in front of camera plane so you don't see things behind you
            //2) it's on the screen (left)
            //3) it's on the screen (right)
            //4) z_buffer, with perpendicular distance
            if (transform_y > 0 && stripe > 0 && stripe < screen_width && transform_y < z_buffer[stripe])
            {
                for (int y = draw_start_y; y < draw_end_y; y++)
                {
                    // 256 and 128 factors to avoid floats
                    int d = (y - v_move_screen) * 256 - screen_height * 128 + sprite_height * 128;
                    int tex_y = ((d * tex_height) / sprite_height) / 256;

                    // Get color from the texture
                    uintrgb *rgb = &textures[sprite.texture][(tex_x + tex_width * tex_y) * 3];

                    // Paint pixel if it isn't black, black is the invisible color
                    if (rgb[0] != 0 && rgb[1] != 0 && rgb[2] != 0)
                    {
                        write_rgb_color(stripe, y, &textures[sprite.texture][(tex_x + tex_width * tex_y) * 3]);
                    }
                }
            }
        }
    }
}

void RaycastWorld::render_mini_map()
{
    // TODO: make this a setting
    usize cell_size = 10;

    // Flip the y-axis when drawing the minimap
    usize origin_y = cell_size * map_height;

    // TODO: make this a setting
    usize x0 = 1, y0 = 1;
    usize xf = map_width - 2, yf = map_height - 2;

    for (usize y = 0; y < map_height; y++)
    {
        for (usize x = 0; x < map_width; x++)
        {
            auto cell = world_map[y][x];
            usize color;
            if (x == x0 && y == y0)
            {
                // Color of home cell
                color = 0x00FF00;
            }
            else if (x == xf && y == yf)
            {
                // Color of destination cell
                color = 0x0000FF;
            }
            else
            {
                // Wall or hall
                color = cell == 0 ? 0xAAAAAA : 0x222222;
            }
            for (usize sy = y * cell_size; sy < y * cell_size + cell_size; ++sy)
            {
                for (usize sx = x * cell_size; sx < x * cell_size + cell_size; ++sx)
                {
                    write_hex_color(sx, origin_y - sy, color);
                }
            }
        }
    }

    // Draw the agent
    usize ar = 2;
    for (int y = pos_y * cell_size - ar; y < pos_y * cell_size + ar; y++)
    {
        for (int x = pos_x * cell_size - ar; x < pos_x * cell_size + ar; x++)
        {
            write_hex_color(x, origin_y - y, 0xFF0000);
        }
    }

    // Draw the agent's direction
    auto dx = (pos_x + dir_x * ar * cell_size) - pos_x;
    auto dy = (pos_y + dir_y * ar * cell_size) - pos_y;
    auto nx = std::max(1, (int)std::abs(dx));
    auto ny = std::max(1, (int)std::abs(dy));
    auto sign_x = dx > 0 ? 1 : -1;
    auto sign_y = dy > 0 ? 1 : -1;

    auto newpx = pos_x * cell_size;
    auto newpy = pos_y * cell_size;

    for (auto ix = 0, iy = 0; ix < nx || iy < ny;)
    {
        if ((0.5 + ix) / nx < (0.5 + iy) / ny)
        {
            // next step is horizontal
            newpx += sign_x;
            ix++;
        }
        else
        {
            // next step is vertical
            newpy += sign_y;
            iy++;
        }
        write_hex_color(std::max(0, (int)newpx), std::max(0, int(origin_y - newpy)), 0xFF0000);
    }
}

#endif // _RAYCAST_WORLD_H_
