//
// Originally by: Spektre on stackoverflow
// https://stackoverflow.com/questions/47239797/ray-casting-with-different-height-size/47251071#47251071
//
// Edited by: Anthony Clark
//

// TODO:
// - switch from vector to array (templated?)
// - minmapping with txr2
// - change PLR to LOC
// - change map from uint32 to byte
// - check for eof when loading scene?
// - time drawing segments
// - review includes
// - review methods and fucntions
// - review variables
// - texture color?
// - move away from ppm
// - move colors and other variables to a config space
// - check divide by zero fix
// - make structs classes
// - move classes and enum inside as appropriate
// - check variable scopes (particularly in draw)
// - rays not all stopping at the same wall?

#ifndef _CASTER_H_
#define _CASTER_H_

using dimension = unsigned int;
using scalar = double;

#include <cstddef>
using std::byte;

#include <fstream>
using std::ifstream;

#include <iostream>
using std::cerr;

#include <string>
using std::string;

#include <vector>
using std::vector;

#include <limits>
using std::numeric_limits;

constexpr scalar dINF = numeric_limits<scalar>::max();

constexpr long double
divide(long double a, long double b)
{
    return b == 0 ? 0 : a / b;
}

constexpr long double deg2rad(long double deg)
{
    return deg * 3.141592 / 180;
}

constexpr long double operator"" _deg(long double deg)
{
    return deg2rad(deg);
}

struct RGB
{
    byte r, g, b;
};

constexpr RGB operator"" _rgb(unsigned long long rgb)
{
    RGB color{
        (byte)((rgb & 0xFF0000) >> 16),
        (byte)((rgb & 0x00FF00) >> 8),
        (byte)((rgb & 0x0000FF))};

    return color;
}

enum class RayType
{
    HORIZONTAL,
    VERTICAL,
    NONE
};

struct Caster
{
    dimension map_xsize, map_ysize;
    vector<uint32_t> map;
    // Each map cell... height?

    dimension can_xsize, can_ysize;
    vector<RGB> canvas;

    dimension tex_xsize, tex_ysize, tex_count;
    vector<byte> texture;

    scalar camx, camy, camz, cama;
    scalar view_angle_rad;
    scalar focal_length_cells;

    bool ready;

    struct Ray
    {
        scalar hitx, hity, length;
        dimension hit_cell_contents;
        RayType type;
    };

    vector<Ray> rays;

    Caster() = delete;
    Caster(string map_filename, string tex_filename, dimension width, dimension height, scalar fov, scalar focal_length);

    void load_scene(string filename);
    void load_texture(string filename);

    void draw();
    void draw_line(long x0, long y0, long x1, long y1, RGB color);
    void update(scalar dt);

    inline dimension cani(dimension x, dimension y)
    {
        return x + y * can_xsize;
    }

    inline dimension mapi(dimension x, dimension y)
    {
        return x + y * map_xsize;
    }
};

Caster::Caster(string map_filename, string tex_filename, dimension width, dimension height, scalar fov, scalar focal_length)
{
    ready = true;

    // Update map and view location
    load_scene(map_filename);

    // Initialize canvas
    can_xsize = width;
    can_ysize = height;
    canvas.resize(can_xsize * can_ysize);

    // Initialize rays
    rays.resize(width);

    // Load textures
    load_texture(tex_filename);

    // Initialize view properties
    view_angle_rad = fov;
    focal_length_cells = focal_length;
}

void Caster::load_scene(string filename)
{
    ifstream map_infile(filename, std::ios::binary);
    if (!map_infile.is_open())
    {
        cerr << "Could not open file: " << filename << "\n";
        ready = false;
        return;
    }

    // Check for the magic header
    constexpr uint32_t MAP_MAGIC = ' PAM';
    uint32_t map_word;
    map_infile.read(reinterpret_cast<char *>(&map_word), sizeof(map_word));

    if (map_word != MAP_MAGIC)
    {
        cerr << "Map file does not start with ' PAM'.\n";
        ready = false;
        return;
    }

    // Set the size of the map
    map_infile.read(reinterpret_cast<char *>(&map_xsize), sizeof(map_xsize));
    map_infile.read(reinterpret_cast<char *>(&map_ysize), sizeof(map_ysize));
    map.resize(map_xsize * map_ysize);

    // Read map data
    auto row_size = map_xsize * sizeof(map.at(0));
    for (dimension row = 0; row < map_ysize; row++)
    {
        map_infile.read(reinterpret_cast<char *>(map.data() + map_xsize * row), row_size);
    }

    // Check for the magic number
    constexpr uint32_t LOC_MAGIC = ' RLP';
    uint32_t loc_word;
    map_infile.read(reinterpret_cast<char *>(&loc_word), sizeof(loc_word));

    if (loc_word != LOC_MAGIC)
    {
        cerr << "Map file does not contain ' COL'.\n";
        ready = false;
        return;
    }

    // Read view position and angle
    map_infile.read(reinterpret_cast<char *>(&camx), sizeof(camx));
    map_infile.read(reinterpret_cast<char *>(&camy), sizeof(camy));
    map_infile.read(reinterpret_cast<char *>(&camz), sizeof(camz));
    map_infile.read(reinterpret_cast<char *>(&cama), sizeof(cama));

    // Check state
    if (!map_infile.good() /*|| map_infile.peek() != map_infile.eof()*/)
    {
        cerr << "Error reading file.\n";
        ready = false;
        return;
    }
}

void Caster::load_texture(string filename)
{
    // Open file and seek to the end
    ifstream tex_infile(filename, std::ios::binary | std::ios::ate);
    if (!tex_infile.is_open())
    {
        cerr << "Could not open file: " << filename << "\n";
        ready = false;
        return;
    }

    // Read the size and then seek to beggining
    std::streamsize file_size = tex_infile.tellg();
    tex_infile.seekg(0, std::ios::beg);

    // Read PPM header
    string header;
    dimension pixel_size;
    char space;
    tex_infile >> header >> tex_xsize >> tex_ysize >> pixel_size >> std::noskipws >> space;

    std::streamsize header_size = tex_infile.tellg();
    auto texture_size = file_size - header_size;

    if (texture_size != tex_xsize * tex_ysize * 3)
    {
        cerr << "Texture size does not match.\n";
        ready = false;
        return;
    }

    // Read pixel data
    texture.resize(texture_size);
    if (!tex_infile.read(reinterpret_cast<char *>(texture.data()), texture_size))
    {
        cerr << "Could not read texture file.\n";
        ready = false;
        return;
    }

    // Set number of textures (assuming square)
    tex_count = tex_xsize / tex_ysize;
    tex_xsize = tex_ysize;
}

void Caster::draw_line(long x0, long y0, long x1, long y1, RGB color)
{
    // https://www.redblobgames.com/grids/line-drawing.html#orthogonal-steps

    auto dx = x1 - x0,
         dy = y1 - y0;
    auto nx = abs((long)dx), ny = abs((long)dy);
    auto sign_x = dx > 0 ? 1 : -1, sign_y = dy > 0 ? 1 : -1;

    // auto p = new Point(x0, y0);
    // auto points = [new Point(p.x, p.y)];

    dimension x = x0, y = y0;
    dimension ix = 0, iy = 0;

    canvas.at(cani(x, y)) = color;

    while (ix < nx || iy < ny)
    {
        if ((0.5 + ix) / nx < (0.5 + iy) / ny)
        {
            // next step is horizontal
            x += sign_x;
            ix++;
        }
        else
        {
            // next step is vertical
            y += sign_y;
            iy++;
        }
        canvas.at(cani(x, y)) = color;
    };
}

void Caster::draw()
{
    dimension can_halfsize = can_xsize * can_ysize / 2;

    // Draw the sky
    auto sky_color = 0x0080FF_rgb;
    fill(canvas.begin(), canvas.begin() + can_halfsize, sky_color);

    // Draw the ground
    auto ground_color = 0x404040_rgb;
    fill(canvas.begin() + can_halfsize, canvas.end(), ground_color);

    // Cast rays
    auto ch = 155.0 + fabs(100.0 * sin(cama));
    auto cv = 155.0 + fabs(100.0 * cos(cama));
    auto ray_angle = cama - (0.5 * view_angle_rad);
    auto da = divide(view_angle_rad, can_xsize - 1);

    for (auto &ray : rays)
    {
        ray.hitx = camx;
        ray.hity = camy;
        ray.hit_cell_contents = 0xFFFFFFFF;
        ray.type = RayType::NONE;
        ray.length = dINF;

        //
        // Vertical hits
        //

        scalar c0 = 0;
        scalar dx0 = cos(ray_angle);

        scalar ll0 = dINF;

        scalar xx0, dy0, yy0, dx, dy;

        if (dx0 < 0.0)
        {
            c0 = 1;
            xx0 = floor(camx) - 0.001;
            dx0 = -1.0;
        }
        if (dx0 > 0.0)
        {
            c0 = 1;
            xx0 = ceil(camx) + 0.001;
            dx0 = +1.0;
        }
        if (c0)
        {
            dy0 = tan(ray_angle);
            yy0 = camy + ((xx0 - camx) * dy0);
            dy0 *= dx0;
            dx = xx0 - camx;
            dy = yy0 - camy;
            ll0 = (dx * dx) + (dy * dy);
        }

        //
        // Horizontal hits
        //

        scalar c1 = 0;
        scalar dy1 = sin(ray_angle);

        scalar ll1 = dINF;

        scalar xx1, dx1, yy1;

        if (dy1 < 0.0)
        {
            c1 = 1;
            yy1 = floor(camy) - 0.001;
            dy1 = -1.0;
        }
        if (dy1 > 0.0)
        {
            c1 = 1;
            yy1 = ceil(camy) + 0.001;
            dy1 = +1.0;
        }
        if (c1)
        {
            dx1 = divide(1.0, tan(ray_angle));
            xx1 = camx + ((yy1 - camy) * dx1);
            dx1 *= dy1;
            dx = xx1 - camx;
            dy = yy1 - camy;
            ll1 = (dx * dx) + (dy * dy);
        }

        //
        //
        //

        dimension height0 = can_ysize;

        bool _hit, _back = false, _bck = true;

        if (!c0)
        {
            ll0 = dINF;
        }
        if (!c1)
        {
            ll1 = dINF;
        }

        //
        //
        //

        int i = 0;
        while (c0 || c1)
        {
            _hit = false;

            // Vertical
            if (c0)
            {
                if (xx0 < 0.0)
                {
                    c0 = 0;
                    ll0 = dINF;
                }
                if (xx0 >= map_xsize)
                {
                    c0 = 0;
                    ll0 = dINF;
                }
                if (yy0 < 0.0)
                {
                    c0 = 0;
                    ll0 = dINF;
                }
                if (yy0 >= map_ysize)
                {
                    c0 = 0;
                    ll0 = dINF;
                }
            }

            if ((c0) && (ll0 < ll1))
            {
                auto m = dimension(xx0 - dx0);
                if ((m >= 0.0) && (m < map_xsize) && (!_bck))
                {
                    auto c = map.at(mapi(m, yy0));
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        ray.hit_cell_contents = c;
                        ray.type = RayType::VERTICAL;
                        ray.length = ll0;
                        ray.hitx = xx0;
                        ray.hity = yy0;
                        _hit = true;
                        _back = true;
                        _bck = true;
                    }
                }
                if (!_hit)
                {
                    auto c = map.at(mapi(xx0, yy0));
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        ray.hit_cell_contents = c;
                        ray.type = RayType::VERTICAL;
                        ray.length = ll0;
                        ray.hitx = xx0;
                        ray.hity = yy0;
                        _hit = true;
                        _back = false;
                        _bck = false;
                    }
                    xx0 += dx0;
                    dx = xx0 - camx;
                    yy0 += dy0;
                    dy = yy0 - camy;
                    ll0 = (dx * dx) + (dy * dy);
                }
            }

            // Horizontal
            if (c1)
            {
                if (xx1 < 0.0)
                {
                    c1 = 0;
                    ll1 = dINF;
                }
                if (xx1 >= map_xsize)
                {
                    c1 = 0;
                    ll1 = dINF;
                }
                if (yy1 < 0.0)
                {
                    c1 = 0;
                    ll1 = dINF;
                }
                if (yy1 >= map_ysize)
                {
                    c1 = 0;
                    ll1 = dINF;
                }
            }

            if ((c1) && (ll0 > ll1) && (!_hit))
            {
                auto m = dimension(yy1 - dy1);
                if ((m >= 0.0) && (m < map_ysize) && (!_bck))
                {
                    auto c = map.at(mapi(xx1, m));
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        ray.hit_cell_contents = c;
                        ray.type = RayType::HORIZONTAL;
                        ray.length = ll1;
                        ray.hitx = xx1;
                        ray.hity = yy1;
                        _hit = true;
                        _back = true;
                        _bck = true;
                    }
                }
                if (!_hit)
                {
                    auto c = map.at(mapi(xx1, yy1));
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        ray.hit_cell_contents = c;
                        ray.type = RayType::HORIZONTAL;
                        ray.length = ll1;
                        ray.hitx = xx1;
                        ray.hity = yy1;
                        _hit = true;
                        _back = false;
                        _bck = false;
                    }
                    xx1 += dx1;
                    dx = xx1 - camx;
                    yy1 += dy1;
                    dy = yy1 - camy;
                    ll1 = (dx * dx) + (dy * dy);
                }
            }
        }

        ray_angle += da;
    }

    //
    // Render Mini Map
    //

    // TODO: ?
    dimension m = 10;
    dimension mx = 10;

    if ((can_xsize >= map_xsize * m) && (can_ysize >= map_ysize * m))
    {
        // Draw walls
        for (dimension y = 0; y < map_ysize * m; y++)
        {
            for (dimension x = 0; x < map_xsize * m; x++)
            {
                RGB color;
                // TODO: simplify condition check
                if ((map.at(mapi(x / m, y / m)) & 0xFFFF) != 0xFFFF)
                {
                    color = 0x00808080_rgb;
                }
                else
                {
                    color = 0x00000000_rgb;
                }
                canvas.at(cani(x, y)) = color;
            }
        }

        // TODO: double?
        scalar x = double(camx * mx); // view rays
        scalar y = double(camy * mx);

        RGB ray_color = 0x00005050_rgb;
        // // TODO: how to translate pmMerge?
        // // scr->Canvas->Pen->Mode = pmMerge;
        for (const auto &ray : rays)
        {
            draw_line(x, y, ray.hitx * mx, ray.hity * mx, ray_color);
        }

        // Draw camera position and direction
        // scr->Canvas->Pen->Mode = pmCopy;
        // c = focus * m;
        // scr->Canvas->Pen->Color = 0x000000FF;
        // scr->Canvas->Brush->Color = 0x000000FF;
        // scr->Canvas->MoveTo(x, y);
        // scr->Canvas->LineTo(DWORD(ray[xs2].x * mx), DWORD(ray[xs2].y * mx));
        // scr->Canvas->Ellipse(x - c, y - c, x + c, y + c);

        // Draw grid lines
        RGB grid_color = 0x00202020_rgb;
        for (y = 0; y <= map_ysize; y++)
        {
            for (x = 0; x <= map_xsize; x++)
            {
                draw_line(0, y * m, map_xsize * m, y * m, grid_color);
                draw_line(x * m, 0, x * m, map_ysize * m, grid_color);
            }
        }

        // Draw selected cell
        // x = keys.mx * m;
        // y = keys.my * m;
        // scr->Canvas->Pen->Color = 0x0020FFFF;
        // scr->Canvas->MoveTo(x, y);
        // scr->Canvas->LineTo(x + m, y);
        // scr->Canvas->LineTo(x + m, y + m);
        // scr->Canvas->LineTo(x, y + m);
        // scr->Canvas->LineTo(x, y);
    }
}

#endif
