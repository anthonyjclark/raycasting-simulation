// TODO:
// - pass in texture info
// - pass in images
// - take into account dt
// - add functions for directly setting pose and camera
// - second constructor that additionally takes pose and camera settings
// - use gpu?
// - automatically call renderview when getBuffer is called?

#if !defined(_RAYCAST_WORLD_H_)
#define _RAYCAST_WORLD_H_

#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>

#include "utilities.h"

// Texture dimensions must be power of two
#define texWidth 64
#define texHeight 64

// Parameters for scaling and moving the sprites
#define uDiv 1
#define vDiv 1
#define vMove 0.0

using usize = uint32_t;
using World = std::vector<std::vector<int>>;
using Tex = std::vector<usize>;
using TexDict = std::unordered_map<usize, std::string>;
using TexMap = std::unordered_map<usize, Tex>;

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

class RaycastWorld
{
private:
    // World map
    usize mapWidth;
    usize mapHeight;
    World worldMap;

    // Render buffer
    usize screenWidth;
    usize screenHeight;
    std::vector<uint8_t> buffer;

    // Write a single color pixel to the buffer
    inline void writeColor(int x, int y, int c)
    {
        buffer[(x * 3 + 0) + y * screenWidth * 3] = (c & 0xFF0000) >> 16;
        buffer[(x * 3 + 1) + y * screenWidth * 3] = (c & 0xFF00) >> 8;
        buffer[(x * 3 + 2) + y * screenWidth * 3] = (c & 0xFF);
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

    // View and camera pose
    double posX, posY, posZ;
    double pitch;
    double dirX, dirY;
    double planeX, planeY;

    // Motion
    Turn turnDirection;
    double turnSpeed;

    Walk walkDirection;
    double walkSpeed;

    bool needToRender;

public:
    RaycastWorld() = delete;
    RaycastWorld(usize width, usize height, World world, TexDict texInfo);

    // void setPosition(double x, double y, double z = 0.0);
    // void setAngle(double a);

    void setTurn(Turn turn) { turnDirection = turn; }
    void setWalk(Walk walk) { walkDirection = walk; }

    void updatePose();
    void renderView();

    auto getBuffer() { return buffer.data(); }
    auto getWidth() { return screenWidth; }
    auto getHeight() { return screenHeight; }

    auto getX() { return posX; }
    auto getY() { return posY; }

    void setPosition(double x, double y, double z = 0)
    {
        posX = x;
        posY = y;
        posZ = z;
        needToRender = true;
    }
    void setDirection(double x, double y)
    {
        dirX = x;
        dirY = y;
        needToRender = true;
    }
};

RaycastWorld::RaycastWorld(usize width, usize height, World world, TexDict texInfo)
    : worldMap(world), screenWidth(width), screenHeight(height), buffer(width * height * 3), ZBuffer(width)
{
    mapHeight = worldMap.size();
    mapWidth = worldMap[0].size();

    // Loop through given texture files
    unsigned long tw, th, error = 0;
    for (auto const &[texNum, texFilename] : texInfo)
    {
        Tex texture(texWidth * texHeight);
        error |= loadImage(texture, tw, th, texFilename);
        textures[texNum] = texture;
    }

    // for (int i = 0; i < 11; i++)
    // {
    //     texture[i].resize(texWidth * texHeight);
    // }

    // // Load some textures
    // error |= loadImage(texture[0], tw, th, "../textures/wood.png");
    // error |= loadImage(texture[1], tw, th, "../textures/redbrick.png");
    // error |= loadImage(texture[2], tw, th, "../textures/redbrick-left.png");
    // error |= loadImage(texture[3], tw, th, "../textures/redbrick-right.png");

    // // Load some sprite textures
    // error |= loadImage(texture[8], tw, th, "../textures/barrel.png");

    if (error)
    {
        std::cerr << "Error loading images." << std::endl;
        return;
    }

    // Location on the map
    posX = 1.5;
    posY = 1.5;

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

        if (worldMap[int(newX)][int(newY)] == 0)
        {
            posX = newX;
            posY = newY;
            needToRender = true;
        }
    }
}

void RaycastWorld::renderView()
{
    if (needToRender)
    {
        // Floor casting
        renderFloorCeiling();

        // Wall casting
        renderWalls();

        // Sprite casting
        // renderSprites();

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
            int tx = (int)(texWidth * (floorX - cellX)) & (texWidth - 1);
            int ty = (int)(texHeight * (floorY - cellY)) & (texHeight - 1);

            floorX += floorStepX;
            floorY += floorStepY;

            // Choose texture and draw the pixel
            int checkerBoardPattern = (int(cellX + cellY)) & 1;
            int floorTexture;
            if (checkerBoardPattern == 0)
                floorTexture = 1;
            else
                floorTexture = 2;
            int ceilingTexture = 0;

            // Floor or ceiling (and make it a bit darker)
            int texNum = isFloor ? floorTexture : ceilingTexture;
            usize color = textures[texNum][texWidth * ty + tx];

            // TODO: temporary hack
            color = isFloor ? (checkerBoardPattern == 0 ? 0x666666 : 0xEEEEEE) : color;

            color = (color >> 1) & 0x7F7F7F;

            writeColor(x, y, color);
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
            hit = worldMap[mapX][mapY] > 0;
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
        int texNum = worldMap[mapX][mapY];

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

            usize color = textures[texNum][texHeight * texY + texX];

            // Darken
            if (side == Hit::Y)
            {
                color = (color >> 1) & 0x7F7F7F;
            }

            writeColor(x, y, color);
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

    // Sort by distance (TODO: reverse order?)
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
                    usize color = textures[sprite.texture][texWidth * texY + texX];

                    // Paint pixel if it isn't black, black is the invisible color
                    if ((color & 0x00FFFFFF) != 0)
                    {
                        writeColor(stripe, y, color);
                    }
                }
            }
        }
    }
}
#endif // _RAYCAST_WORLD_H_
