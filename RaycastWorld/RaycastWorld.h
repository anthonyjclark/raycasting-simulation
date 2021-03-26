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
const usize texWidth = 1024;
const usize texHeight = 1024;

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

enum Turn
{
    LEFT = -1,
    STOP = 0,
    RIGHT = 1,
};

enum Walk
{
    BACKWARD = -1,
    STOPPED = 0,
    FORWARD = 1,
};

double deg2rad(double deg)
{
    return deg * 3.1415926 / 180.0;
}

void readPNG(std::vector<uintrgb> &out, const std::string &filename)
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
    usize mapWidth;
    usize mapHeight;
    World worldMap;
    bool showMiniMap;

    // Render buffer
    usize screenWidth;
    usize screenHeight;
    std::vector<uintrgb> buffer;

    // Write a single color pixel to the buffer
    inline void writeColorHex(int x, int y, usize c, bool darken = false)
    {
        c = darken ? (c >> 1) & 0x7F : c;
        buffer[(x * 3 + 0) + y * screenWidth * 3] = (c & 0xFF0000) >> 16;
        buffer[(x * 3 + 1) + y * screenWidth * 3] = (c & 0xFF00) >> 8;
        buffer[(x * 3 + 2) + y * screenWidth * 3] = (c & 0xFF);
    }

    inline void writeColorRGB(int x, int y, uintrgb *c, bool darken = false)
    {
        buffer[(x * 3 + 0) + y * screenWidth * 3] = darken ? (c[0] >> 1) & 0x7F : c[0];
        buffer[(x * 3 + 1) + y * screenWidth * 3] = darken ? (c[1] >> 1) & 0x7F : c[1];
        buffer[(x * 3 + 2) + y * screenWidth * 3] = darken ? (c[2] >> 1) & 0x7F : c[2];
    }

    // Texture buffers
    TexMap textures;

    // 1D Zbuffer for sprites
    std::vector<double> ZBuffer;

    // Arrays used to sort the sprites
    std::vector<Sprite> sprites;

    void renderFloorCeiling();
    void renderWalls();
    void renderSprites();
    void renderMiniMap();

    // View and camera pose
    double posX, posY, posZ;
    double pitch;
    double dirX, dirY;
    double planeX, planeY;

    // Initial view
    double initialX, initialY, initialAngle;

    // Motion
    Turn turnDirection;
    double turnSpeed;

    Walk walkDirection;
    double walkSpeed;

    bool needToRender;

public:
    RaycastWorld() = delete;
    RaycastWorld(usize width, usize height, std::string mazeFilePath);

    void updatePose();
    void renderView();

    void setTurn(Turn turn) { turnDirection = turn; }
    void setWalk(Walk walk) { walkDirection = walk; }

    auto getWidth() { return screenWidth; }
    auto getHeight() { return screenHeight; }

    auto getX() { return posX; }
    auto getY() { return posY; }

    auto getDirX() { return dirX; }
    auto getDirY() { return dirY; }

    auto getTurnSpeed() { return turnSpeed; }
    auto getWalkSpeed() { return walkSpeed; }

    void setX(double x)
    {
        posX = x;
        needToRender = true;
    }

    void setY(double y)
    {
        posY = y;
        needToRender = true;
    }

    void setZ(double z)
    {
        posZ = z;
        needToRender = true;
    }

    void setPosition(double x, double y, double z = 0.0)
    {
        posX = x;
        posY = y;
        posZ = z;
        needToRender = true;
    }

    void setDirection(double radians, double fov = 1.152)
    {
        dirX = cos(radians);
        dirY = sin(radians);

        // 2 * atan(L/1.0) = fov
        // L = tan(fov/2)
        double planeLength = tan(fov / 2);
        planeX = planeLength * sin(radians);
        planeY = -planeLength * cos(radians);

        needToRender = true;
    }

    void reset()
    {
        setPosition(initialX, initialY);
        setDirection(initialAngle);
    }

    void toggleMiniMap()
    {
        showMiniMap = !showMiniMap;
        needToRender = true;
    }

    auto getBuffer()
    {
        renderView();
        return buffer.data();
    }

    void savePNG(std::string filename)
    {
        renderView();
        auto error = lodepng::encode(filename, buffer, screenWidth, screenHeight, LCT_RGB, 8);
        if (error)
        {
            std::cerr << "Error writing PNG " << error << ": " << lodepng_error_text(error) << std::endl;
        }
    }

    void addSprite(double x, double y, usize texID)
    {
        Sprite sp{x, y, texID, 1, 0.0};
        sprites.push_back(sp);
    }
};

RaycastWorld::RaycastWorld(usize width, usize height, std::string mazeFilePath)
    : screenWidth(width), screenHeight(height), buffer(width * height * 3), ZBuffer(width)
{
    std::ifstream mazeFile(mazeFilePath);
    if (!mazeFile.is_open())
    {
        std::cerr << "Error opening maze file." << std::endl;
        return;
    }

    usize numTextures = 0;
    mazeFile >> numTextures;
    std::cout << "Number of texture: " << numTextures << std::endl;

    // Loop through given texture files
    for (usize texID = 0; texID < numTextures; ++texID)
    {
        std::string texFilename;
        mazeFile >> texFilename;
        std::cout << "Texture filename: " << texFilename << std::endl;

        // Tex texture(texWidth * texHeight * 3);
        Tex texture; // TODO: does this need to be pre-sized?
        readPNG(texture, texFilename);
        textures[texID] = texture;
    }

    mazeFile >> mapWidth >> mapHeight;
    std::cout << "Map is " << mapWidth << " by " << mapHeight << std::endl;

    for (usize row = 0; row < mapHeight; ++row)
    {
        std::vector<int> mapRow;
        for (usize col = 0; col < mapWidth; ++col)
        {
            int cell_value;
            mazeFile >> cell_value;
            mapRow.push_back(cell_value);
        }
        worldMap.emplace_back(mapRow);
    }

    // Map needs to be reversed so that the bottom left is 0,0
    std::reverse(worldMap.begin(), worldMap.end());

    showMiniMap = false;

    // TODO: add sprites to maze file?
    // Add the sprites
    // double x, y;
    // usize texID;
    // while (mazeFile >> x >> y >> texID)
    // {
    //     addSprite(x, y, texID);
    //     break;
    // }

    // Location on the map
    std::string direction;
    mazeFile >> posX >> posY >> direction;

    // Position into middle of cell
    posX += 0.5;
    posY += 0.5;

    initialX = posX;
    initialY = posY;

    // Vertical camera strafing up/down, for jumping/crouching.
    // 0 means standard height.
    // Expressed in screen pixels a wall at distance 1 shifts
    posZ = 0;

    // Looking up/down, expressed in screen pixels the horizon shifts
    pitch = 0;

    // Direction vector
    dirX = -1.0;
    dirY = 0.0;

    // The 2d raycaster version of camera plane
    planeX = 0.0;
    planeY = 0.66;

    // Motion and rendering
    turnDirection = STOP;
    turnSpeed = 2.5 * (3.1415926 / 180);
    walkDirection = STOPPED;
    walkSpeed = 0.05;

    // Set direction
    if (direction == "Dir.NORTH")
    {
        initialAngle = deg2rad(90);
    }
    else if (direction == "Dir.EAST")
    {
        initialAngle = deg2rad(0);
    }
    else if (direction == "Dir.SOUTH")
    {
        initialAngle = deg2rad(270);
    }
    else
    {
        initialAngle = deg2rad(180);
    }
    setDirection(initialAngle);

    std::cout << "Initial pose: "
              << initialX << ", " << initialY << " "
              << initialAngle << " (" << direction << ")" << std::endl;

    // Loop through maze file to get coordinates of the goal
    double curX, curY;
    std::string curDir;
    while (mazeFile >> curX >> curY >> curDir)
        ;
    std::cout << "Goal position: "
              << curX << ", " << curY << std::endl;

    needToRender = true;
    renderView();
}

void RaycastWorld::updatePose()
{
    if (turnDirection != STOP)
    {
        double rotSpeed = turnDirection * turnSpeed;
        double oldDirX = dirX;
        dirX = oldDirX * cos(-rotSpeed) - dirY * sin(-rotSpeed);
        dirY = oldDirX * sin(-rotSpeed) + dirY * cos(-rotSpeed);

        double oldPlaneX = planeX;
        planeX = oldPlaneX * cos(-rotSpeed) - planeY * sin(-rotSpeed);
        planeY = oldPlaneX * sin(-rotSpeed) + planeY * cos(-rotSpeed);

        needToRender = true;
    }

    if (walkDirection != STOPPED)
    {
        double step = walkDirection * walkSpeed;
        double newX = posX + dirX * step;
        double newY = posY + dirY * step;

        // TODO: x vs y?

        if (worldMap[int(posY)][int(newX)] == 0)
        {
            posX = newX;
            needToRender = true;
        }

        if (worldMap[int(newY)][int(posX)] == 0)
        {
            posY = newY;
            needToRender = true;
        }
    }
}

void RaycastWorld::renderView()
{
    if (needToRender)
    {
        renderFloorCeiling();
        renderWalls();
        renderSprites();

        if (showMiniMap)
        {
            renderMiniMap();
        }
        needToRender = false;
    }
}

void RaycastWorld::renderFloorCeiling()
{
    for (auto y = 0u; y < screenHeight; ++y)
    {
        // Whether this section is floor or ceiling
        bool isFloor = y > screenHeight / 2 + pitch;

        // rayDir for leftmost ray (x = 0) and rightmost ray (x = screenWidth)
        float rayDirX0 = dirX - planeX;
        float rayDirY0 = dirY - planeY;
        float rayDirX1 = dirX + planeX;
        float rayDirY1 = dirY + planeY;

        // Current y position compared to the center of the screen (the horizon)
        int p = isFloor ? (y - screenHeight / 2 - pitch) : (screenHeight / 2 - y + pitch);

        // Vertical position of the camera.
        // NOTE: with 0.5, it's exactly in the center between floor and ceiling,
        // matching also how the walls are being raycasted. For different values
        // than 0.5, a separate loop must be done for ceiling and floor since
        // they're no longer symmetrical.
        float camZ = isFloor ? (0.5 * screenHeight + posZ) : (0.5 * screenHeight - posZ);

        // Horizontal distance from the camera to the floor for the current row.
        // 0.5 is the z position exactly in the middle between floor and ceiling.
        // NOTE: this is affine texture mapping, which is not perspective correct
        // except for perfectly horizontal and vertical surfaces like the floor.
        // NOTE: this formula is explained as follows: The camera ray goes through
        // the following two points: the camera itself, which is at a certain
        // height (posZ), and a point in front of the camera (through an imagined
        // vertical plane containing the screen pixels) with horizontal distance
        // 1 from the camera, and vertical position p lower than posZ (posZ - p). When going
        // through that point, the line has vertically traveled by p units and
        // horizontally by 1 unit. To hit the floor, it instead needs to travel by
        // posZ units. It will travel the same ratio horizontally. The ratio was
        // 1 / p for going through the camera plane, so to go posZ times farther
        // to reach the floor, we get that the total horizontal distance is posZ / p.
        float rowDistance = camZ / p;

        // calculate the real world step vector we have to add for each x (parallel to camera plane)
        // adding step by step avoids multiplications with a weight in the inner loop
        float floorStepX = rowDistance * (rayDirX1 - rayDirX0) / screenWidth;
        float floorStepY = rowDistance * (rayDirY1 - rayDirY0) / screenWidth;

        // real world coordinates of the leftmost column. This will be updated as we step to the right.
        float floorX = posX + rowDistance * rayDirX0;
        float floorY = posY + rowDistance * rayDirY0;

        for (auto x = 0u; x < screenWidth; ++x)
        {
            // The cell coord is simply got from the integer parts of floorX and floorY
            int cellX = floorX;
            int cellY = floorY;

            // Get the texture coordinate from the fractional part
            int texX = (int)(texWidth * (floorX - cellX)) & (texWidth - 1);
            int texY = (int)(texHeight * (floorY - cellY)) & (texHeight - 1);

            floorX += floorStepX;
            floorY += floorStepY;

            // Choose texture and draw the pixel
            // int checkerBoardPattern = (int(cellX + cellY)) & 1;
            // int floorTexture = checkerBoardPattern == 0 ? 1 : 2;
            // int ceilingTexture = 0;

            // // Floor or ceiling (and make it a bit darker)
            // if (isFloor)
            // {
            //     auto color = checkerBoardPattern == 0 ? 0x666666 : 0xEEEEEE;
            //     writeColorHex(x, y, (color >> 1) & 0x7F7F7F);
            // }
            // else
            // {
            //     int texID = ceilingTexture;
            //     writeColorRGB(x, y, &textures[texID][(texX + texWidth * texY) * 3]);
            // }
            int texID = isFloor ? 0 : 1;
            writeColorRGB(x, y, &textures[texID][(texX + texWidth * texY) * 3]);
        }
    }
}

void RaycastWorld::renderWalls()
{
    for (auto x = 0u; x < screenWidth; x++)
    {
        // x-coordinate in camera space
        double cameraX = 2 * x / double(screenWidth) - 1;

        // Calculate ray position and direction
        double rayDirX = dirX + planeX * cameraX;
        double rayDirY = dirY + planeY * cameraX;

        // Which box of the map we're in
        int mapX = posX;
        int mapY = posY;

        // Length of ray from one x or y-side to next x or y-side
        double deltaDistX = std::abs(1 / rayDirX);
        double deltaDistY = std::abs(1 / rayDirY);
        double perpWallDist;

        // What direction to step in x or y-direction (either +1 or -1)
        int stepX = rayDirX < 0 ? -1 : 1;
        int stepY = rayDirY < 0 ? -1 : 1;

        // Length of ray from current position to next x or y-side
        double sideDistX = rayDirX < 0 ? (posX - mapX) * deltaDistX : (mapX + 1.0 - posX) * deltaDistX;
        double sideDistY = rayDirY < 0 ? (posY - mapY) * deltaDistY : (mapY + 1.0 - posY) * deltaDistY;

        // Was a NS or a EW wall hit?
        Hit side;

        // Perform DDA
        bool hit = false;
        while (hit == false)
        {
            // Jump to next map square, OR in x-direction, OR in y-direction
            if (sideDistX < sideDistY)
            {
                sideDistX += deltaDistX;
                mapX += stepX;
                side = Hit::X;
            }
            else
            {
                sideDistY += deltaDistY;
                mapY += stepY;
                side = Hit::Y;
            }

            // Check if ray has hit a wall
            hit = worldMap[mapY][mapX] > 0;
        }

        // Calculate distance of perpendicular ray (Euclidean distance will give fisheye effect!)
        if (side == Hit::X)
        {
            perpWallDist = (mapX - posX + (1 - stepX) / 2) / rayDirX;
        }
        else
        {
            perpWallDist = (mapY - posY + (1 - stepY) / 2) / rayDirY;
        }

        // Calculate height of line to draw on screen
        int lineHeight = screenHeight / perpWallDist;

        // Calculate lowest and highest pixel to fill in current stripe
        int drawStart = -lineHeight / 2 + screenHeight / 2 + pitch + (posZ / perpWallDist);
        drawStart = std::max(drawStart, 0);

        int drawEnd = lineHeight / 2 + screenHeight / 2 + pitch + (posZ / perpWallDist);
        drawEnd = std::min(drawEnd, static_cast<int>(screenHeight) - 1);

        // Texturing calculations (1 subtracted from it so that texture 0 can be used!)
        int texID = worldMap[mapY][mapX];

        // Calculate value of wallX (where exactly the wall was hit)
        double wallX = side == Hit::X ? posY + perpWallDist * rayDirY : posX + perpWallDist * rayDirX;
        wallX -= floor(wallX);

        // x-coordinate on the texture
        int texX = int(wallX * double(texWidth));
        if ((side == Hit::X && rayDirX > 0) || (side == Hit::Y && rayDirY < 0))
        {
            texX = texWidth - texX - 1;
        }

        // TODO: an integer-only bresenham or DDA like algorithm could make the texture coordinate stepping faster

        // How much to increase the texture coordinate per screen pixel
        double step = 1.0 * texHeight / lineHeight;

        // Starting texture coordinate
        double texPos = (drawStart - pitch - (posZ / perpWallDist) - screenHeight / 2 + lineHeight / 2) * step;

        for (int y = drawStart; y < drawEnd; y++)
        {
            // Cast the texture coordinate to integer, and mask with (texHeight - 1) in case of overflow
            int texY = (int)texPos & (texHeight - 1);
            texPos += step;
            writeColorRGB(x, y, &textures[texID][(texX + texWidth * texY) * 3], side == Hit::Y);
        }

        // Set zbuffer as distance to wall for sprite casting
        ZBuffer[x] = perpWallDist;
    }
}

void RaycastWorld::renderSprites()
{
    // Compute distances
    for (auto &sprite : sprites)
    {
        sprite.distance2 = (posX - sprite.x) * (posX - sprite.x) + (posY - sprite.y) * (posY - sprite.y);
    }

    // Sort by distance
    std::sort(sprites.begin(), sprites.end(), [](auto a, auto b) {
        return a.distance2 > b.distance2;
    });

    // Project and draw sprites
    // for (int i = 0; i < numSprites; i++)
    for (const auto &sprite : sprites)
    {
        // Translate sprite position to relative to camera
        double spriteX = sprite.x - posX;
        double spriteY = sprite.y - posY;

        // Transform sprite with the inverse camera matrix
        // [ planeX   dirX ] -1                                  [ dirY      -dirX ]
        // [               ]    =  1/(planeX*dirY-dirX*planeY) * [                 ]
        // [ planeY   dirY ]                                     [ -planeY  planeX ]

        // Required for correct matrix multiplication
        double invDet = 1.0 / (planeX * dirY - dirX * planeY);

        double transformX = invDet * (dirY * spriteX - dirX * spriteY);
        // This is actually the depth inside the screen, that what Z is in 3D, the distance of
        // sprite to player, matching sqrt(spriteDistance[i])
        double transformY = invDet * (-planeY * spriteX + planeX * spriteY);

        int spriteScreenX = (screenWidth / 2) * (1 + transformX / transformY);

        int vMoveScreen = int(vMove / transformY) + pitch + posZ / transformY;

        // Calculate height of the sprite on screen
        // Using "transformY" instead of the real distance prevents fisheye
        int spriteHeight = abs(int(screenHeight / (transformY))) / vDiv;

        // Calculate lowest and highest pixel to fill in current stripe
        int drawStartY = -spriteHeight / 2 + screenHeight / 2 + vMoveScreen;
        drawStartY = std::max(drawStartY, 0);

        int drawEndY = spriteHeight / 2 + screenHeight / 2 + vMoveScreen;
        drawEndY = std::min(drawEndY, static_cast<int>(screenHeight) - 1);

        // Calculate width of the sprite
        int spriteWidth = abs(int(screenHeight / (transformY))) / uDiv;

        usize drawStartX = std::max(-spriteWidth / 2 + spriteScreenX, 0);
        usize drawEndX = std::max(std::min(spriteWidth / 2 + spriteScreenX, static_cast<int>(screenWidth) - 1), 0);

        // Loop through every vertical stripe of the sprite on screen
        for (usize stripe = drawStartX; stripe < drawEndX; stripe++)
        {
            int texX = int(256 * (stripe - (-spriteWidth / 2 + spriteScreenX)) * texWidth / spriteWidth) / 256;

            //the conditions in the if are:
            //1) it's in front of camera plane so you don't see things behind you
            //2) it's on the screen (left)
            //3) it's on the screen (right)
            //4) ZBuffer, with perpendicular distance
            if (transformY > 0 && stripe > 0 && stripe < screenWidth && transformY < ZBuffer[stripe])
            {
                for (int y = drawStartY; y < drawEndY; y++)
                {
                    // 256 and 128 factors to avoid floats
                    int d = (y - vMoveScreen) * 256 - screenHeight * 128 + spriteHeight * 128;
                    int texY = ((d * texHeight) / spriteHeight) / 256;

                    // Get color from the texture
                    uintrgb *rgb = &textures[sprite.texture][(texX + texWidth * texY) * 3];

                    // Paint pixel if it isn't black, black is the invisible color
                    if (rgb[0] != 0 && rgb[1] != 0 && rgb[2] != 0)
                    {
                        writeColorRGB(stripe, y, &textures[sprite.texture][(texX + texWidth * texY) * 3]);
                    }
                }
            }
        }
    }
}

void RaycastWorld::renderMiniMap()
{
    // TODO: make this a setting
    usize cellSize = 10;

    // Flip the y-axis when drawing the minimap
    usize originY = cellSize * mapHeight;

    // TODO: make this a setting
    usize x0 = 1, y0 = 1;
    usize xf = mapWidth - 2, yf = mapHeight - 2;

    for (usize y = 0; y < mapHeight; y++)
    {
        for (usize x = 0; x < mapWidth; x++)
        {
            auto cell = worldMap[y][x];
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
            for (usize sy = y * cellSize; sy < y * cellSize + cellSize; ++sy)
            {
                for (usize sx = x * cellSize; sx < x * cellSize + cellSize; ++sx)
                {
                    writeColorHex(sx, originY - sy, color);
                }
            }
        }
    }

    // Draw the agent
    usize ar = 2;
    for (int y = posY * cellSize - ar; y < posY * cellSize + ar; y++)
    {
        for (int x = posX * cellSize - ar; x < posX * cellSize + ar; x++)
        {
            writeColorHex(x, originY - y, 0xFF0000);
        }
    }

    // Draw the agent's direction
    auto dx = (posX + dirX * ar * cellSize) - posX;
    auto dy = (posY + dirY * ar * cellSize) - posY;
    auto nx = std::max(1, (int)std::abs(dx));
    auto ny = std::max(1, (int)std::abs(dy));
    auto sign_x = dx > 0 ? 1 : -1;
    auto sign_y = dy > 0 ? 1 : -1;

    auto newpx = posX * cellSize;
    auto newpy = posY * cellSize;

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
        writeColorHex(std::max(0, (int)newpx), std::max(0, int(originY - newpy)), 0xFF0000);
    }
}

#endif // _RAYCAST_WORLD_H_
