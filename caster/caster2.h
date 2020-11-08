#if !defined(_CASTER_H_)
#define _CASTER_H_

#include <cmath>
#include <string>
#include <vector>
#include <iostream>

#include "utilities.h"

#define screenWidth 640
#define screenHeight 480
#define texWidth 64  // must be power of two
#define texHeight 64 // must be power of two
#define mapWidth 24
#define mapHeight 24

#define numSprites 19

//parameters for scaling and moving the sprites
#define uDiv 1
#define vDiv 1
#define vMove 0.0

struct Sprite
{
    double x;
    double y;
    int texture;
};

//sort the sprites based on distance
void sortSprites(int *order, double *dist, int amount)
{
    std::vector<std::pair<double, int>> sprites(amount);
    for (int i = 0; i < amount; i++)
    {
        sprites[i].first = dist[i];
        sprites[i].second = order[i];
    }
    std::sort(sprites.begin(), sprites.end());
    // restore in reverse order to go from farthest to nearest
    for (int i = 0; i < amount; i++)
    {
        dist[i] = sprites[amount - i - 1].first;
        order[i] = sprites[amount - i - 1].second;
    }
}

class Caster
{
    // TODO: private
    // private:
public:
    int worldMap[mapWidth][mapHeight] =
        {{8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 6, 4, 4, 6, 4, 6, 4, 4, 4, 6, 4},
         {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4},
         {8, 0, 3, 3, 0, 0, 0, 0, 0, 8, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6},
         {8, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6},
         {8, 0, 3, 3, 0, 0, 0, 0, 0, 8, 8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4},
         {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 4, 0, 0, 0, 0, 0, 6, 6, 6, 0, 6, 4, 6},
         {8, 8, 8, 8, 0, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 6, 0, 0, 0, 0, 0, 6},
         {7, 7, 7, 7, 0, 7, 7, 7, 7, 0, 8, 0, 8, 0, 8, 0, 8, 4, 0, 4, 0, 6, 0, 6},
         {7, 7, 0, 0, 0, 0, 0, 0, 7, 8, 0, 8, 0, 8, 0, 8, 8, 6, 0, 0, 0, 0, 0, 6},
         {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 0, 0, 0, 0, 0, 4},
         {7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 6, 0, 6, 0, 6, 0, 6},
         {7, 7, 0, 0, 0, 0, 0, 0, 7, 8, 0, 8, 0, 8, 0, 8, 8, 6, 4, 6, 0, 6, 6, 6},
         {7, 7, 7, 7, 0, 7, 7, 7, 7, 8, 8, 4, 0, 6, 8, 4, 8, 3, 3, 3, 0, 3, 3, 3},
         {2, 2, 2, 2, 0, 2, 2, 2, 2, 4, 6, 4, 0, 0, 6, 0, 6, 3, 0, 0, 0, 0, 0, 3},
         {2, 2, 0, 0, 0, 0, 0, 2, 2, 4, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 3},
         {2, 0, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0, 4, 3, 0, 0, 0, 0, 0, 3},
         {1, 0, 0, 0, 0, 0, 0, 0, 1, 4, 4, 4, 4, 4, 6, 0, 6, 3, 3, 0, 0, 0, 3, 3},
         {2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 6, 6, 0, 0, 5, 0, 5, 0, 5},
         {2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 5, 0, 5, 0, 0, 0, 5, 5},
         {2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 5, 0, 5, 0, 5, 0, 5, 0, 5},
         {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5},
         {2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 2, 5, 0, 5, 0, 5, 0, 5, 0, 5},
         {2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 0, 5, 0, 5, 0, 0, 0, 5, 5},
         {2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5}};
    Sprite sprite[numSprites] =
        {
            {20.5, 11.5, 10}, //green light in front of playerstart
            //green lights in every room
            {18.5, 4.5, 10},
            {10.0, 4.5, 10},
            {10.0, 12.5, 10},
            {3.5, 6.5, 10},
            {3.5, 20.5, 10},
            {3.5, 14.5, 10},
            {14.5, 20.5, 10},

            //row of pillars in front of wall: fisheye test
            {18.5, 10.5, 9},
            {18.5, 11.5, 9},
            {18.5, 12.5, 9},

            //some barrels around the map
            {21.5, 1.5, 8},
            {15.5, 1.5, 8},
            {16.0, 1.8, 8},
            {16.2, 1.2, 8},
            {3.5, 2.5, 8},
            {9.5, 15.5, 8},
            {10.0, 15.1, 8},
            {10.5, 15.8, 8},
    };

    // uint32_t buffer[screenHeight][screenWidth]; // y-coordinate first because it works per scanline
    std::vector<uint32_t> buffer;

    //1D Zbuffer
    double ZBuffer[screenWidth];

    //arrays used to sort the sprites
    int spriteOrder[numSprites];
    double spriteDistance[numSprites];

public:
    Caster(/* args */);
};

Caster::Caster(/* args */) : buffer(screenWidth * screenHeight)
{
    double posX = 22.0, posY = 11.5;    //x and y start position
    double dirX = -1.0, dirY = 1.0;     //initial direction vector
    double planeX = 1.0, planeY = 0.66; //the 2d raycaster version of camera plane
    double pitch = 0;                   // looking up/down, expressed in screen pixels the horizon shifts
    double posZ = 0;                    // vertical camera strafing up/down, for jumping/crouching. 0 means standard height. Expressed in screen pixels a wall at distance 1 shifts

    std::vector<uint32_t> texture[11];
    for (int i = 0; i < 11; i++)
        texture[i].resize(texWidth * texHeight);

    //load some textures
    unsigned long tw, th, error = 0;
    error |= loadImage(texture[0], tw, th, "../textures/eagle.png");
    error |= loadImage(texture[1], tw, th, "../textures/redbrick.png");
    error |= loadImage(texture[2], tw, th, "../textures/purplestone.png");
    error |= loadImage(texture[3], tw, th, "../textures/greystone.png");
    error |= loadImage(texture[4], tw, th, "../textures/bluestone.png");
    error |= loadImage(texture[5], tw, th, "../textures/mossy.png");
    error |= loadImage(texture[6], tw, th, "../textures/wood.png");
    error |= loadImage(texture[7], tw, th, "../textures/colorstone.png");
    if (error)
    {
        std::cerr << "error loading images" << std::endl;
        return;
    }

    //load some sprite textures
    error |= loadImage(texture[8], tw, th, "../textures/barrel.png");
    error |= loadImage(texture[9], tw, th, "../textures/pillar.png");
    error |= loadImage(texture[10], tw, th, "../textures/greenlight.png");
    if (error)
    {
        std::cerr << "error loading images" << std::endl;
        return;
    }

    //FLOOR CASTING
    for (int y = 0; y < screenHeight; ++y)
    {
        // whether this section is floor or ceiling
        bool is_floor = y > screenHeight / 2 + pitch;

        // rayDir for leftmost ray (x = 0) and rightmost ray (x = screenWidth)
        float rayDirX0 = dirX - planeX;
        float rayDirY0 = dirY - planeY;
        float rayDirX1 = dirX + planeX;
        float rayDirY1 = dirY + planeY;

        // Current y position compared to the center of the screen (the horizon)
        int p = is_floor ? (y - screenHeight / 2 - pitch) : (screenHeight / 2 - y + pitch);

        // Vertical position of the camera.
        // NOTE: with 0.5, it's exactly in the center between floor and ceiling,
        // matching also how the walls are being raycasted. For different values
        // than 0.5, a separate loop must be done for ceiling and floor since
        // they're no longer symmetrical.
        float camZ = is_floor ? (0.5 * screenHeight + posZ) : (0.5 * screenHeight - posZ);

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

        for (int x = 0; x < screenWidth; ++x)
        {
            // the cell coord is simply got from the integer parts of floorX and floorY
            int cellX = (int)(floorX);
            int cellY = (int)(floorY);

            // get the texture coordinate from the fractional part
            int tx = (int)(texWidth * (floorX - cellX)) & (texWidth - 1);
            int ty = (int)(texHeight * (floorY - cellY)) & (texHeight - 1);

            floorX += floorStepX;
            floorY += floorStepY;

            // choose texture and draw the pixel
            int checkerBoardPattern = (int(cellX + cellY)) & 1;
            int floorTexture;
            if (checkerBoardPattern == 0)
                floorTexture = 3;
            else
                floorTexture = 4;
            int ceilingTexture = 6;
            uint32_t color;

            if (is_floor)
            {
                // floor
                color = texture[floorTexture][texWidth * ty + tx];
                color = (color >> 1) & 8355711; // make a bit darker
                // buffer[y][x] = color;
                buffer[x + y * screenWidth] = color;
            }
            else
            {
                //ceiling
                color = texture[ceilingTexture][texWidth * ty + tx];
                color = (color >> 1) & 8355711; // make a bit darker
                // buffer[y][x] = color;
                buffer[x + y * screenWidth] = color;
            }
        }
    }

    // WALL CASTING
    for (int x = 0; x < screenWidth; x++)
    {
        //calculate ray position and direction
        double cameraX = 2 * x / double(screenWidth) - 1; //x-coordinate in camera space
        double rayDirX = dirX + planeX * cameraX;
        double rayDirY = dirY + planeY * cameraX;

        //which box of the map we're in
        int mapX = int(posX);
        int mapY = int(posY);

        //length of ray from current position to next x or y-side
        double sideDistX;
        double sideDistY;

        //length of ray from one x or y-side to next x or y-side
        double deltaDistX = std::abs(1 / rayDirX);
        double deltaDistY = std::abs(1 / rayDirY);
        double perpWallDist;

        //what direction to step in x or y-direction (either +1 or -1)
        int stepX;
        int stepY;

        int hit = 0; //was there a wall hit?
        int side;    //was a NS or a EW wall hit?

        //calculate step and initial sideDist
        if (rayDirX < 0)
        {
            stepX = -1;
            sideDistX = (posX - mapX) * deltaDistX;
        }
        else
        {
            stepX = 1;
            sideDistX = (mapX + 1.0 - posX) * deltaDistX;
        }
        if (rayDirY < 0)
        {
            stepY = -1;
            sideDistY = (posY - mapY) * deltaDistY;
        }
        else
        {
            stepY = 1;
            sideDistY = (mapY + 1.0 - posY) * deltaDistY;
        }
        //perform DDA
        while (hit == 0)
        {
            //jump to next map square, OR in x-direction, OR in y-direction
            if (sideDistX < sideDistY)
            {
                sideDistX += deltaDistX;
                mapX += stepX;
                side = 0;
            }
            else
            {
                sideDistY += deltaDistY;
                mapY += stepY;
                side = 1;
            }
            //Check if ray has hit a wall
            if (worldMap[mapX][mapY] > 0)
                hit = 1;
        }

        //Calculate distance of perpendicular ray (Euclidean distance will give fisheye effect!)
        if (side == 0)
            perpWallDist = (mapX - posX + (1 - stepX) / 2) / rayDirX;
        else
            perpWallDist = (mapY - posY + (1 - stepY) / 2) / rayDirY;

        //Calculate height of line to draw on screen
        int lineHeight = (int)(screenHeight / perpWallDist);

        //calculate lowest and highest pixel to fill in current stripe
        int drawStart = -lineHeight / 2 + screenHeight / 2 + pitch + (posZ / perpWallDist);
        if (drawStart < 0)
            drawStart = 0;
        int drawEnd = lineHeight / 2 + screenHeight / 2 + pitch + (posZ / perpWallDist);
        if (drawEnd >= screenHeight)
            drawEnd = screenHeight - 1;
        //texturing calculations
        int texNum = worldMap[mapX][mapY] - 1; //1 subtracted from it so that texture 0 can be used!

        //calculate value of wallX
        double wallX; //where exactly the wall was hit
        if (side == 0)
            wallX = posY + perpWallDist * rayDirY;
        else
            wallX = posX + perpWallDist * rayDirX;
        wallX -= floor((wallX));

        //x coordinate on the texture
        int texX = int(wallX * double(texWidth));
        if (side == 0 && rayDirX > 0)
            texX = texWidth - texX - 1;
        if (side == 1 && rayDirY < 0)
            texX = texWidth - texX - 1;

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
            uint32_t color = texture[texNum][texHeight * texY + texX];
            //make color darker for y-sides: R, G and B byte each divided through two with a "shift" and an "and"
            if (side == 1)
                color = (color >> 1) & 8355711;
            // buffer[y][x] = color;
            buffer[x + y * screenWidth] = color;
        }

        //SET THE ZBUFFER FOR THE SPRITE CASTING
        ZBuffer[x] = perpWallDist; //perpendicular distance is used
    }

    //SPRITE CASTING
    //sort sprites from far to close
    for (int i = 0; i < numSprites; i++)
    {
        spriteOrder[i] = i;
        spriteDistance[i] = ((posX - sprite[i].x) * (posX - sprite[i].x) + (posY - sprite[i].y) * (posY - sprite[i].y)); //sqrt not taken, unneeded
    }
    sortSprites(spriteOrder, spriteDistance, numSprites);

    //after sorting the sprites, do the projection and draw them
    for (int i = 0; i < numSprites; i++)
    {
        //translate sprite position to relative to camera
        double spriteX = sprite[spriteOrder[i]].x - posX;
        double spriteY = sprite[spriteOrder[i]].y - posY;

        //transform sprite with the inverse camera matrix
        // [ planeX   dirX ] -1                                       [ dirY      -dirX ]
        // [               ]       =  1/(planeX*dirY-dirX*planeY) *   [                 ]
        // [ planeY   dirY ]                                          [ -planeY  planeX ]

        double invDet = 1.0 / (planeX * dirY - dirX * planeY); //required for correct matrix multiplication

        double transformX = invDet * (dirY * spriteX - dirX * spriteY);
        double transformY = invDet * (-planeY * spriteX + planeX * spriteY); //this is actually the depth inside the screen, that what Z is in 3D, the distance of sprite to player, matching sqrt(spriteDistance[i])

        int spriteScreenX = int((screenWidth / 2) * (1 + transformX / transformY));

        int vMoveScreen = int(vMove / transformY) + pitch + posZ / transformY;

        //calculate height of the sprite on screen
        int spriteHeight = abs(int(screenHeight / (transformY))) / vDiv; //using "transformY" instead of the real distance prevents fisheye
        //calculate lowest and highest pixel to fill in current stripe
        int drawStartY = -spriteHeight / 2 + screenHeight / 2 + vMoveScreen;
        if (drawStartY < 0)
            drawStartY = 0;
        int drawEndY = spriteHeight / 2 + screenHeight / 2 + vMoveScreen;
        if (drawEndY >= screenHeight)
            drawEndY = screenHeight - 1;

        //calculate width of the sprite
        int spriteWidth = abs(int(screenHeight / (transformY))) / uDiv;
        int drawStartX = -spriteWidth / 2 + spriteScreenX;
        if (drawStartX < 0)
            drawStartX = 0;
        int drawEndX = spriteWidth / 2 + spriteScreenX;
        if (drawEndX >= screenWidth)
            drawEndX = screenWidth - 1;

        //loop through every vertical stripe of the sprite on screen
        for (int stripe = drawStartX; stripe < drawEndX; stripe++)
        {
            int texX = int(256 * (stripe - (-spriteWidth / 2 + spriteScreenX)) * texWidth / spriteWidth) / 256;
            //the conditions in the if are:
            //1) it's in front of camera plane so you don't see things behind you
            //2) it's on the screen (left)
            //3) it's on the screen (right)
            //4) ZBuffer, with perpendicular distance
            if (transformY > 0 && stripe > 0 && stripe < screenWidth && transformY < ZBuffer[stripe])
                for (int y = drawStartY; y < drawEndY; y++) //for every pixel of the current stripe
                {
                    int d = (y - vMoveScreen) * 256 - screenHeight * 128 + spriteHeight * 128; //256 and 128 factors to avoid floats
                    int texY = ((d * texHeight) / spriteHeight) / 256;
                    uint32_t color = texture[sprite[spriteOrder[i]].texture][texWidth * texY + texX]; //get current color from the texture
                    if ((color & 0x00FFFFFF) != 0)
                        // buffer[y][stripe] = color; //paint pixel if it isn't black, black is the invisible color
                        buffer[stripe + y * screenWidth] = color;
                }
        }
    }
}

#endif // _CASTER_H_
