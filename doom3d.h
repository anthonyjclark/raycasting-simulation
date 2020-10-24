#ifndef _DOOM3D_H_
#define _DOOM3D_H_

// TODO:
// 1. check for TODO
// 2. remove casts? might not need them for non-pointer types
// 3. switch from array to vector or Array

// Doom 3D engine ver: 1.000
// Originally by: Spektre on stackoverflow
// https://stackoverflow.com/questions/47239797/ray-casting-with-different-height-size/47251071#47251071
//
// Edited by: Anthony Clark

#include <cmath>

#include <string>
using std::string;

#include <fstream>
using std::ifstream;
using std::ofstream;

#include <iostream>
using std::cerr;

using uint = unsigned int;

constexpr long double deg2rad(long double deg)
{
    return deg * 3.141592 / 180;
}
constexpr long double operator"" _deg(long double deg)
{
    return deg2rad(deg);
}

const uint _Doom3D_cell_size = 10;  // 2D map cell size
const uint _Doom3D_wall_size = 100; // full height of wall in map
#define _Doom3D_filter_txr

class Doom3D
{
public:
    uint mxs, mys, **pmap; // 2D map   // txr + height<<16
    uint sxs, sys, **pscr; // pseudo 3D screen
    // Graphics::TBitmap *scr;
    uint8_t *canvas;
    uint txs, tys, **ptxr, tn; // 2D textures
    // Graphics::TBitmap *txr, *txr2;  // textures, texture mipmaps resolution: /2 and /4
    uint8_t *texture;
    double plrx, plry, plrz, plra; // player position [x,y,z,angle]
    double view_ang;               // [rad] view angle
    double focus;                  // [cells] view focal length
    struct _ray
    {
        double x, y, l; // hit or end of map position
        uint hit;       // map cell of hit or 0xFFFFFFFF
        char typ;       // H/V
        _ray(){};
        _ray(_ray &a) { *this = a; }
        ~_ray(){};
        _ray *operator=(const _ray *a)
        {
            *this = *a;
            return this;
        }
        //_ray* operator = (const _ray &a) { ..copy... return this; }
    };
    _ray *ray; // ray[sxs]

    // keytab keys;
    uint txr_sel;
    uint cell_h;

    Doom3D();
    Doom3D(Doom3D &a) { *this = a; }
    ~Doom3D();
    Doom3D *operator=(const Doom3D *a)
    {
        *this = *a;
        return this;
    }
    //Doom3D* operator = (const Doom3D &a) { ..copy... return this; }

    void map_resize(uint xs, uint ys); // change map resolution
    void map_height(uint height);      // set height for whole map to convert maps from Wolfenstein3D demo
    void map_clear();                  // clear whole map
    void map_save(string name);
    void map_load(string name);
    void scr_resize(uint xs, uint ys);
    void txr_load(string name);

    void draw();
    void update(double dt);
    // void mouse(double x, double y, TShiftState sh)
    // {
    //     x = floor(x / _Doom3D_cell_size);
    //     if (x >= mxs)
    //         x = mxs - 1;
    //     if (x < 0)
    //         x = 0;
    //     y = floor(y / _Doom3D_cell_size);
    //     if (y >= mys)
    //         y = mys - 1;
    //     if (y < 0)
    //         y = 0;
    //     uint xx = x, yy = y;
    //     keys.setm(x, y, sh);
    //     if (keys.Shift.Contains(ssLeft))
    //         pmap[yy][xx] = (txr_sel) | (cell_h << 16);
    //     if (keys.Shift.Contains(ssRight))
    //         pmap[yy][xx] = 0xFFFFFFFF;
    //     keys.rfsmouse();
    // }
    // void wheel(int delta, TShiftState sh)
    // {
    //     if (sh.Contains(ssShift))
    //     {
    //         if (delta < 0)
    //         {
    //             cell_h -= 10;
    //             if (cell_h < 10)
    //                 cell_h = 10;
    //         }
    //         if (delta > 0)
    //         {
    //             cell_h += 10;
    //             if (cell_h > _Doom3D_wall_size)
    //                 cell_h = _Doom3D_wall_size;
    //         }
    //     }
    //     else
    //     {
    //         if (delta < 0)
    //         {
    //             txr_sel--;
    //             if (txr_sel == 0xFFFFFFFF)
    //                 txr_sel = tn - 1;
    //         }
    //         if (delta > 0)
    //         {
    //             txr_sel++;
    //             if (txr_sel == tn)
    //                 txr_sel = 0;
    //         }
    //     }
    // }
};
//---------------------------------------------------------------------------
Doom3D::Doom3D()
{
    mxs = 0;
    mys = 0;
    pmap = nullptr;
    sxs = 0;
    sys = 0;
    // scr = new Graphics::TBitmap;
    canvas = nullptr;
    pscr = nullptr;
    ray = nullptr;
    txs = 0;
    tys = 0;
    // txr = new Graphics::TBitmap;
    texture = nullptr;
    ptxr = nullptr;
    tn = 0;
    // txr2 = new Graphics::TBitmap;
    plrx = 0.0;
    plry = 0.0;
    plrz = 0.0;
    plra = 0.0;
    view_ang = 60.0_deg;
    focus = 0.25;
    txr_sel = 0;
    cell_h = _Doom3D_wall_size;

    txr_load("textures/textures128x128.ppm");
    map_resize(16, 16);
    map_load("Doom3D_map.dat");
}
//---------------------------------------------------------------------------
Doom3D::~Doom3D()
{
    uint y;
    map_save("Doom3D_map.dat");
    if (pmap)
    {
        for (y = 0; y < mys; y++)
            delete[] pmap[y];
        delete[] pmap;
        pmap = nullptr;
    }
    if (ray)
        delete[] ray;
    ray = nullptr;
    if (pscr)
    {
        delete[] pscr;
        pscr = nullptr;
    }
    // if (scr)
    //     delete scr;
    // scr = nullptr;
    if (canvas)
    {
        delete[] canvas;
    }
    canvas = nullptr;
    if (ptxr)
    {
        delete[] ptxr;
        ptxr = nullptr;
    }
    // if (txr)
    //     delete txr;
    // txr = nullptr;
    if (texture)
    {
        delete[] texture;
    }
    texture = nullptr;
    // if (txr2)
    //     delete txr2;
    // txr2 = nullptr;
}
//---------------------------------------------------------------------------
void Doom3D::map_resize(uint xs, uint ys)
{
    uint y;
    if (pmap)
    {
        for (y = 0; y < mys; y++)
            delete[] pmap[y];
        delete[] pmap;
        pmap = nullptr;
    }
    mys = ys;
    mxs = xs;
    pmap = new uint *[mys];
    for (y = 0; y < mys; y++)
        pmap[y] = new uint[mxs];
    map_clear();
    plrx = (mxs - 1) * 0.5;
    plry = (mys - 1) * 0.5;
    plrz = 0.0;
    plra = 0.0_deg;
}
//---------------------------------------------------------------------------
void Doom3D::map_height(uint h)
{
    uint x, y, c;
    for (y = 0; y < mys; y++)
        for (x = 0; x < mxs; x++)
        {
            c = pmap[y][x];
            c &= 0xFFFF;
            c |= h << 16;
            pmap[y][x] = c;
        }
}
//---------------------------------------------------------------------------
void Doom3D::map_clear()
{
    uint x, y, c;
    for (y = 0; y < mys; y++)
        for (x = 0; x < mxs; x++)
        {
            c = 0xFFFFFFFF;
            if ((x == 0) || (x == mxs - 1))
                c = 0;
            if ((y == 0) || (y == mys - 1))
                c = 0;
            pmap[y][x] = c;
        }
}
//---------------------------------------------------------------------------
void Doom3D::map_save(string name)
{
    // int hnd = FileCreate(name);
    // if (hnd < 0)
    //     return;
    ofstream map_outfile(name, std::ios::binary);
    if (!map_outfile.is_open())
    {
        cerr << "Could not open file: " << name << "\n";
        return;
    }
    uint y;
    y = ' PAM';
    // FileWrite(hnd, &y, 4);    // id
    map_outfile.write(reinterpret_cast<char *>(&y), sizeof(y));
    // FileWrite(hnd, &mxs, 4);  // x resolution
    map_outfile.write(reinterpret_cast<char *>(&mxs), sizeof(mxs));
    // FileWrite(hnd, &mys, 4);  // y resolution
    map_outfile.write(reinterpret_cast<char *>(&mys), sizeof(mys));
    // for (y = 0; y < mys; y++) // map
    //     FileWrite(hnd, pmap[y], mxs << 2);
    for (y = 0; y < mys; y++)
    {
        map_outfile.write(reinterpret_cast<char *>(pmap[y]), sizeof(pmap[y][0]) * mxs);
    }
    y = ' RLP';
    // FileWrite(hnd, &y, 4); // id
    map_outfile.write(reinterpret_cast<char *>(&y), sizeof(y));
    // FileWrite(hnd, &plrx, 8);
    map_outfile.write(reinterpret_cast<char *>(&plrx), sizeof(plrx));
    // FileWrite(hnd, &plry, 8);
    map_outfile.write(reinterpret_cast<char *>(&plry), sizeof(plry));
    // FileWrite(hnd, &plrz, 8);
    map_outfile.write(reinterpret_cast<char *>(&plrz), sizeof(plrz));
    // FileWrite(hnd, &plra, 8);
    map_outfile.write(reinterpret_cast<char *>(&plra), sizeof(plra));
    // FileClose(hnd);
}
//---------------------------------------------------------------------------
void Doom3D::map_load(string name)
{
    // int hnd = FileOpen(name, fmOpenRead);
    // if (hnd < 0)
    //     return;
    ifstream map_infile(name, std::ios::binary);
    if (!map_infile.is_open())
    {
        cerr << "Could not open file: " << name << "\n";
        return;
    }
    uint x, y;
    y = ' PAM';
    // FileRead(hnd, &x, 4); // id
    map_infile.read(reinterpret_cast<char *>(&x), sizeof(uint));
    if (x == y)
    {
        // FileRead(hnd, &x, 4); // x resolution
        map_infile.read(reinterpret_cast<char *>(&x), sizeof(x));
        // FileRead(hnd, &y, 4); // y resolution
        map_infile.read(reinterpret_cast<char *>(&y), sizeof(y));
        map_resize(x, y);
        // for (y = 0; y < mys; y++) // map
        //     FileRead(hnd, pmap[y], mxs << 2);
        for (y = 0; y < mys; y++)
        {
            map_infile.read(reinterpret_cast<char *>(pmap[y]), sizeof(pmap[y][0]) * mxs);
        }
    }
    y = ' RLP';
    // FileRead(hnd, &x, 4); // id
    map_infile.read(reinterpret_cast<char *>(&x), sizeof(x));
    if (x == y)
    {
        // FileRead(hnd, &plrx, 8);
        map_infile.read(reinterpret_cast<char *>(&plrx), sizeof(plrx));
        // FileRead(hnd, &plry, 8);
        map_infile.read(reinterpret_cast<char *>(&plry), sizeof(plry));
        // FileRead(hnd, &plrz, 8);
        map_infile.read(reinterpret_cast<char *>(&plrz), sizeof(plrz));
        // FileRead(hnd, &plra, 8);
        map_infile.read(reinterpret_cast<char *>(&plra), sizeof(plra));
    }
    // FileClose(hnd);
}
//---------------------------------------------------------------------------
void Doom3D::scr_resize(uint xs, uint ys)
{
    // scr->HandleType = bmDIB;
    // scr->PixelFormat = pf32bit;
    // scr->SetSize(xs, ys);
    // sxs = scr->Width;
    // sys = scr->Height;

    // width BY height BY channels (RGB)
    canvas = new uint8_t[xs * ys * 3];
    sxs = xs;
    sys = ys;

    delete[] pscr;
    pscr = new uint *[sys];
    for (uint y = 0; y < sys; y++)
    {
        // pscr[y] = (uint *)scr->ScanLine[y];
        // TODO: is this valid?
        pscr[y] = reinterpret_cast<uint *>(canvas + y * sxs);
    }
    if (ray)
    {
        delete[] ray;
    }
    ray = new _ray[sxs];
}
//---------------------------------------------------------------------------
void Doom3D::txr_load(string name)
{
    // string ext = ExtractFileExt(name).LowerCase();
    // for (;;)
    // {
    //     if (ext == ".bmp")
    //     {
    //         txr->LoadFromFile(name);
    //         break;
    //     }
    //     if (ext == ".jpg")
    //     {
    //         TJPEGImage *jpg = new TJPEGImage;
    //         if (jpg == nullptr)
    //             return;
    //         jpg->LoadFromFile(name);
    //         txr->Assign(jpg);
    //         delete jpg;
    //         break;
    //     }
    //     return;
    // }
    // uint y = tys;
    // txr->HandleType = bmDIB;
    // txr->PixelFormat = pf32bit;
    // txs = txr->Width;
    // tys = txr->Height;

    // Open file, seek to the end, read the size, seek to beggining
    ifstream texture_file(name, std::ios::binary | std::ios::ate);
    std::streamsize texture_size = texture_file.tellg();
    texture_file.seekg(0, std::ios::beg);

    // Read PPM header
    string header;
    int sizex, sizey, sizec;
    texture_file >> header >> sizex >> sizey >> sizec;

    std::cout << "Texture size : " << sizex * sizey * 3 << " (" << sizex << "x" << sizey << "x3)\n";
    std::cout << "Expecting    : " << 128 * 128 * 48 * 3 << "\n";

    texture = new uint8_t[texture_size];
    if (!texture_file.read(reinterpret_cast<char *>(texture), texture_size))
    {
        cerr << "Could not read texture file.\n";
    }

    txs = sizex;
    tys = sizey;

    // // mip map
    // txr2->SetSize(txs >> 1, (tys >> 1) + (tys >> 2));
    // txr2->Canvas->StretchDraw(TRect(0, 0, txs >> 1, tys >> 1), txr);
    // txr2->Canvas->StretchDraw(TRect(0, tys >> 1, txs >> 2, (tys >> 1) + (tys >> 2)), txr);
    tn = txs / tys;
    txs = tys;
    delete[] ptxr;
    ptxr = new uint *[tys];
    // for (y = 0; y < tys; y++)
    //     ptxr[y] = (uint *)txr->ScanLine[y];
    for (int y = 0; y < tys; y++)
    {
        ptxr[y] = reinterpret_cast<uint *>(texture + y * txs);
    }
}
//---------------------------------------------------------------------------
void Doom3D::draw()
{
    // total time measurement
    // tbeg();
    // double tperf0 = performance_tms;

    string tcls, tray, tmap, ttotal;
    double a, a0, da, dx, dy, l, mx, my;
    uint x, y, xs2, ys2, c, m;
    double xx0, yy0, dx0, dy0, ll0;
    uint c0, d0;
    double xx1, yy1, dx1, dy1, ll1;
    uint c1, d1;
    _ray *p;
    xs2 = sxs >> 1;
    ys2 = sys >> 1;

    // aspect ratio,view angle corrections
    a = 90.0_deg - view_ang;
    double wall = double(sxs) * (1.25 + (0.288 * a) + (2.04 * a * a)); // [px]

    // floor,ceilling/sky
    // tbeg();
    for (y = 0; y < ys2; y++)
        for (x = 0; x < sxs; x++)
            pscr[y][x] = 0x000080FF;
    for (; y < sys; y++)
        for (x = 0; x < sxs; x++)
            pscr[y][x] = 0x00404040;
    // tend();
    // tcls = tstr(1) + " cls";

    // [cast rays]
    // tbeg();
    // diffuse + ambient lighting
    uint ch = 155.0 + fabs(100.0 * sin(plra));
    uint cv = 155.0 + fabs(100.0 * cos(plra));
    a0 = plra - (0.5 * view_ang);
    // da = divide(view_ang, sxs - 1);
    // TODO: is this the correct way to handle divide by 0 here?
    da = (sxs - 1) == 0 ? 0 : view_ang / (sxs - 1);
    mx = mxs;
    my = mys;
    for (p = ray, a = a0, x = 0; x < sxs; x++, a += da, p++)
    {
        p->x = plrx;
        p->y = plry;
        p->hit = 0xFFFFFFFF;
        p->typ = ' ';
        p->l = 1.0e20;
        ll0 = ll1 = p->l;
        // grid V-line hits
        c0 = 0;
        dx0 = cos(a);
        if (dx0 < 0.0)
        {
            c0 = 1;
            xx0 = floor(plrx) - 0.001;
            dx0 = -1.0;
        }
        if (dx0 > 0.0)
        {
            c0 = 1;
            xx0 = ceil(plrx) + 0.001;
            dx0 = +1.0;
        }
        if (c0)
        {
            dy0 = tan(a);
            yy0 = plry + ((xx0 - plrx) * dy0);
            dy0 *= dx0;
            dx = xx0 - plrx;
            dy = yy0 - plry;
            ll0 = (dx * dx) + (dy * dy);
        }
        // grid H-line hits
        c1 = 0;
        dy1 = sin(a);
        if (dy1 < 0.0)
        {
            c1 = 1;
            yy1 = floor(plry) - 0.001;
            dy1 = -1.0;
        }
        if (dy1 > 0.0)
        {
            c1 = 1;
            yy1 = ceil(plry) + 0.001;
            dy1 = +1.0;
        }
        if (c1)
        {
            // dx1 = divide(1.0, tan(a));
            // TODO: divide by 0?
            dx1 = tan(a) == 0 ? 0 : 1.0 / tan(a);
            xx1 = plrx + ((yy1 - plry) * dx1);
            dx1 *= dy1;
            dx = xx1 - plrx;
            dy = yy1 - plry;
            ll1 = (dx * dx) + (dy * dy);
        }
        int height0 = sys; // already rendered height [pixels]
        bool _hit, _back = false, _bck = true;
        if (!c0)
            ll0 = 1e20;
        if (!c1)
            ll1 = 1e20;
        for (; c0 || c1;)
        {
            _hit = false;
            // grid V-line hits
            if (c0)
            {
                if (xx0 < 0.0)
                {
                    c0 = 0;
                    ll0 = 1e20;
                }
                if (xx0 >= mx)
                {
                    c0 = 0;
                    ll0 = 1e20;
                }
                if (yy0 < 0.0)
                {
                    c0 = 0;
                    ll0 = 1e20;
                }
                if (yy0 >= my)
                {
                    c0 = 0;
                    ll0 = 1e20;
                }
            }
            if ((c0) && (ll0 < ll1))
            {
                m = uint(xx0 - dx0);
                if ((m >= 0.0) && (m < mxs) && (!_bck))
                {
                    c = pmap[uint(yy0)][m];
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        p->hit = c;
                        p->typ = 'V';
                        p->l = ll0;
                        p->x = xx0;
                        p->y = yy0;
                        _hit = true;
                        _back = true;
                        _bck = true;
                    }
                }
                if (!_hit)
                {
                    c = pmap[uint(yy0)][uint(xx0)];
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        p->hit = c;
                        p->typ = 'V';
                        p->l = ll0;
                        p->x = xx0;
                        p->y = yy0;
                        _hit = true;
                        _back = false;
                        _bck = false;
                    }
                    xx0 += dx0;
                    dx = xx0 - plrx;
                    yy0 += dy0;
                    dy = yy0 - plry;
                    ll0 = (dx * dx) + (dy * dy);
                }
            }
            // grid H-line hits
            if (c1)
            {
                if (xx1 < 0.0)
                {
                    c1 = 0;
                    ll1 = 1e20;
                }
                if (xx1 >= mx)
                {
                    c1 = 0;
                    ll1 = 1e20;
                }
                if (yy1 < 0.0)
                {
                    c1 = 0;
                    ll1 = 1e20;
                }
                if (yy1 >= my)
                {
                    c1 = 0;
                    ll1 = 1e20;
                }
            }
            if ((c1) && (ll0 > ll1) && (!_hit))
            {
                m = uint(yy1 - dy1);
                if ((m >= 0.0) && (m < mys) && (!_bck))
                {
                    c = pmap[m][uint(xx1)];
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        p->hit = c;
                        p->typ = 'H';
                        p->l = ll1;
                        p->x = xx1;
                        p->y = yy1;
                        _hit = true;
                        _back = true;
                        _bck = true;
                    }
                }
                if (!_hit)
                {
                    c = pmap[uint(yy1)][uint(xx1)];
                    if ((c & 0xFFFF) != 0xFFFF)
                    {
                        p->hit = c;
                        p->typ = 'H';
                        p->l = ll1;
                        p->x = xx1;
                        p->y = yy1;
                        _hit = true;
                        _back = false;
                        _bck = false;
                    }
                    xx1 += dx1;
                    dx = xx1 - plrx;
                    yy1 += dy1;
                    dy = yy1 - plry;
                    ll1 = (dx * dx) + (dy * dy);
                }
            }
            // render scan line
            if (_hit)
            {
                union
                {
                    uint dd;
                    uint8_t db[4];
                } cc;
                int tx, ty, sy, sy0, sy1, cnt, dsy, dty;
                p->l = sqrt(p->l) * cos(a - plra); // anti fish eye
                // m = divide(wall * focus, p->l);    // projected wall half size
                // TODO: divide by 0
                m = p->l == 0 ? 0 : (wall * focus) / p->l;
                c = 0;
                if (p->typ == 'H')
                {
                    c = ch;
                    tx = double(double(txs) * (p->x - floor(p->x)));
                }
                if (p->typ == 'V')
                {
                    c = cv;
                    tx = double(double(txs) * (p->y - floor(p->y)));
                }
                tx += txs * (p->hit & 0xFFFF);

                // prepare interpolation
                sy1 = ys2 + m;
                //              sy0=ys2-m;                                          // constant wall height
                sy0 = sy1 - (((m + m) * (p->hit >> 16)) / _Doom3D_wall_size); // variable wall height
                dty = tys - 1;
                dsy = sy1 - sy0 + 1;
                // skip sy>=sys
                if (sy1 >= sys)
                    sy1 = sys - 1;
                // skip sy<0
                for (cnt = dsy, sy = sy0, ty = 0; sy < 0; sy++)
                {
                    cnt -= dty;
                    while (cnt <= 0)
                    {
                        cnt += dsy;
                        ty++;
                    }
                }

#ifdef _Doom3D_filter_txr
                uint r = 0, g = 0, b = 0, n = 0;
#else
                cc.dd = ptxr[ty][tx];
                cc.db[0] = uint((uint(cc.db[0]) * c) >> 8);
                cc.db[1] = uint((uint(cc.db[1]) * c) >> 8);
                cc.db[2] = uint((uint(cc.db[2]) * c) >> 8);
#endif
                // continue sy>=0
                y = height0;
                if (sy1 > height0)
                    sy1 = height0;
                if (sy0 < height0)
                    height0 = sy0;
                if (_back)
                {
                    for (sy = sy0; sy <= y; sy++)
                    {
                        if ((sy > 0) && (sy < sys))
                            pscr[sy][x] = 0x0000FF00;
                    }
                }
                else
                    for (; sy <= sy1; sy++)
                    {
#ifdef _Doom3D_filter_txr
                        if (!n)
                        {
                            cc.dd = ptxr[ty][tx];
                            b += uint(cc.db[0]);
                            g += uint(cc.db[1]);
                            r += uint(cc.db[2]);
                            n += 256;
                        }
                        if ((sy > 0) && (sy < sys))
                        {
                            cc.db[0] = uint(c * b / n);
                            b = 0;
                            cc.db[1] = uint(c * g / n);
                            g = 0;
                            cc.db[2] = uint(c * r / n);
                            r = 0;
                            n = 0;
                            pscr[sy][x] = cc.dd;
                        }
                        cnt -= dty;
                        while (cnt <= 0)
                        {
                            cnt += dsy;
                            ty++;
                            cc.dd = ptxr[ty][tx];
                            b += uint(cc.db[0]);
                            g += uint(cc.db[1]);
                            r += uint(cc.db[2]);
                            n += 256;
                        }
#else
                        if ((sy > 0) && (sy < sys))
                            pscr[sy][x] = cc.dd;
                        cnt -= dty;
                        while (cnt <= 0)
                        {
                            cnt += dsy;
                            ty++;
                            cc.dd = ptxr[ty][tx];
                            cc.db[0] = uint((uint(cc.db[0]) * c) >> 8);
                            cc.db[1] = uint((uint(cc.db[1]) * c) >> 8);
                            cc.db[2] = uint((uint(cc.db[2]) * c) >> 8);
                        }
#endif
                    }
                if (height0 < 0)
                    break;
            }
        }
    }
    // tend();
    // tray = tstr(1) + " ray";

    // [2D map]
    // tbeg();
    m = _Doom3D_cell_size;
    mx = _Doom3D_cell_size;
    if ((sxs >= mxs * m) && (sys >= mys * m))
    {
        for (y = 0; y < mys * m; y++) // pmap[][]
            for (x = 0; x < mxs * m; x++)
            {
                if ((pmap[y / m][x / m] & 0xFFFF) != 0xFFFF)
                    c = 0x00808080;
                else
                    c = 0x00000000;
                pscr[y][x] = c;
            }
        x = double(plrx * mx); // view rays
        y = double(plry * mx);
        // scr->Canvas->Pen->Color = 0x00005050;
        uint8_t r = 0, g = 0x50, b = 0x50;
        // scr->Canvas->Pen->Mode = pmMerge;
        for (c = 0; c < sxs; c++)
        {
            // scr->Canvas->MoveTo(x, y);
            // scr->Canvas->LineTo(uint(ray[c].x * mx), uint(ray[c].y * mx));
            // Grid Walking: https://www.redblobgames.com/grids/line-drawing.html#stepping
            // int dx = uint(ray[c].x * mx) - x;
            // int dy = uint(ray[c].y * mx) - y;
            // int nx = dx > 0 ? dx : -dx;
            // int ny = dy > 0 ? dy : -dy;
            // int sign_x = dx > 0 ? 1 : -1;
            // int sign_y = dy > 0 ? 1 : -1;
            // int current_x = x;
            // int current_y = y;
            // for (int ix = 0, iy = 0; ix < nx || iy < ny;)
            // {
            //     if ((0.5 + ix) / nx < (0.5 + iy) / ny)
            //     {
            //         // next step is horizontal
            //         current_x += sign_x;
            //         ix++;
            //     }
            //     else
            //     {
            //         // next step is vertical
            //         current_y += sign_y;
            //         iy++;
            //     }
            //     canvas[current_x + current_y * sxs + 0] = r;
            //     canvas[current_x + current_y * sxs + 1] = g;
            //     canvas[current_x + current_y * sxs + 2] = b;
            //     std::cout << current_x << ", " << current_y << "\n";
            // }
        }
        // scr->Canvas->Pen->Mode = pmCopy;
        c = focus * m; // player and view direction
        // scr->Canvas->Pen->Color = 0x000000FF;
        // scr->Canvas->Brush->Color = 0x000000FF;
        // scr->Canvas->MoveTo(x, y);
        // scr->Canvas->LineTo(uint(ray[xs2].x * mx), uint(ray[xs2].y * mx));
        // scr->Canvas->Ellipse(x - c, y - c, x + c, y + c);
        // scr->Canvas->Pen->Color = 0x00202020;
        for (y = 0; y <= mys; y++) // map grid
            for (x = 0; x <= mxs; x++)
            {
                // scr->Canvas->MoveTo(0, y * m);
                // scr->Canvas->LineTo(mxs * m, y * m);
                // scr->Canvas->MoveTo(x * m, 0);
                // scr->Canvas->LineTo(x * m, mys * m);
            }
        // x = keys.mx * m; // selected cell
        // y = keys.my * m;
        // scr->Canvas->Pen->Color = 0x0020FFFF;
        // scr->Canvas->MoveTo(x, y);
        // scr->Canvas->LineTo(x + m, y);
        // scr->Canvas->LineTo(x + m, y + m);
        // scr->Canvas->LineTo(x, y + m);
        // scr->Canvas->LineTo(x, y);
    }
    // tend();
    // tmap = tstr(1) + " map";

    // [editor]
    if (txr_sel != 0xFFFFFFFF)
    {
        int x = sxs, y = 5, s0, s1, s2, i, j;
        s0 = txs >> 1;
        s1 = txs >> 2;
        s2 = (s0 * cell_h) / _Doom3D_wall_size;

        for (i = -3; i <= 3; i++)
        {
            j = txr_sel + i;
            while (j < 0)
                j += tn;
            while (j >= tn)
                j -= tn;
            if (i)
            {
                // scr->Canvas->CopyRect(TRect(x - s1, y + (s1 >> 1), x, s1 + (s1 >> 1)), txr2->Canvas, TRect(s1 * j, s0, s1 * j + s1, s0 + s1));
                x -= s1 + 5;
            }
            else
            {
                // scr->Canvas->CopyRect(TRect(x - s0, y + s0 - s2, x, s0), txr2->Canvas, TRect(s0 * j, 0, s0 * j + s0, s2));
                x -= s0 + 5;
            }
        }
    }

    // total time measurement
    // performance_tms = tperf0;
    // tend();
    // ttotal = tstr(1) + " total";

    x = m * mxs + m;
    c = 16;
    y = -c;
    // scr->Canvas->Font->Color = clYellow;
    // scr->Canvas->Brush->Style = bsClear;
    // scr->Canvas->TextOutA(x, y += c, string().sprintf("player: %.2lf x %.2lf x %.2lf", plrx, plry, plrz));
    // scr->Canvas->TextOutA(x, y += c, string().sprintf(" mouse: %.2lf x %.2lf", keys.mx, keys.my));
    // scr->Canvas->TextOutA(x, y += c, tray);
    // scr->Canvas->TextOutA(x, y += c, tcls);
    // scr->Canvas->TextOutA(x, y += c, tmap);
    // scr->Canvas->TextOutA(x, y += c, ttotal);
    // scr->Canvas->TextOutA(x, y += c, string().sprintf("   key: %d", keys.Key));

    // aspect ratio test
    /*
    c=ys2*7/10;
    scr->Canvas->Rectangle(xs2-c,ys2-c,xs2+c,ys2+c);
*/
    // cross
    // c = 4, m = 32;
    // scr->Canvas->Pen->Color = clRed;
    // scr->Canvas->MoveTo(xs2 - c, ys2 - m);
    // scr->Canvas->LineTo(xs2 - c, ys2 - c);
    // scr->Canvas->LineTo(xs2 - m, ys2 - c);
    // scr->Canvas->MoveTo(xs2 + c, ys2 - m);
    // scr->Canvas->LineTo(xs2 + c, ys2 - c);
    // scr->Canvas->LineTo(xs2 + m, ys2 - c);
    // scr->Canvas->MoveTo(xs2 - c, ys2 + m);
    // scr->Canvas->LineTo(xs2 - c, ys2 + c);
    // scr->Canvas->LineTo(xs2 - m, ys2 + c);
    // scr->Canvas->MoveTo(xs2 + c, ys2 + m);
    // scr->Canvas->LineTo(xs2 + c, ys2 + c);
    // scr->Canvas->LineTo(xs2 + m, ys2 + c);

    // scr->Canvas->Brush->Style = bsSolid;
}
//---------------------------------------------------------------------------
void Doom3D::update(double dt)
{
    int move = 0;
    double da = 120.0_deg * dt;
    double dl = 5.0 * dt;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    // if (keys.get(104))
    // {
    //     plra -= da;
    //     if (plra < 0.0)
    //         plra += pi2;
    // } // turn l/r
    // if (keys.get(105))
    // {
    //     plra += da;
    //     if (plra >= pi2)
    //         plra -= pi2;
    // }
    // if (keys.get(101))
    // {
    //     move = 1;
    //     dx = +dl * cos(plra);
    //     dy = +dl * sin(plra);
    // } // move f/b
    // if (keys.get(98))
    // {
    //     move = 1;
    //     dx = -dl * cos(plra);
    //     dy = -dl * sin(plra);
    // }
    // if (keys.get(102))
    // {
    //     move = 1;
    //     dx = dl * cos(plra - 90_deg);
    //     dy = dl * sin(plra - 90_deg);
    // } // strafe l/r
    // if (keys.get(99))
    // {
    //     move = 1;
    //     dx = dl * cos(plra + 90_deg);
    //     dy = dl * sin(plra + 90_deg);
    // }
    // if (keys.get(100))
    // {
    //     move = 1;
    //     dz = +dl;
    // } // strafe u/d
    // if (keys.get(97))
    // {
    //     move = 1;
    //     dz = -dl;
    // }
    if (move) // update/test plr position
    {
        double x, y, z, mx, my;
        x = plrx + dx;
        mx = mxs - focus;
        y = plry + dy;
        my = mys - focus;
        z = plrz + dz;
        if ((z >= 0.0) && (z <= _Doom3D_wall_size))
            plrz = z;
        ;
        if (x < focus)
            x = focus;
        if (x > mx)
            x = mx;
        if (y < focus)
            y = focus;
        if (y > my)
            y = my;
        // dx *= divide(focus, dl);
        // TODO: divide by 0
        dx *= dl == 0 ? 0 : focus / dl;
        // dy *= divide(focus, dl);
        // TODO: divide by 0
        dy *= dl == 0 ? 0 : focus / dl;
        if ((pmap[uint(y + dy)][uint(x + dx)] & 0xFFFF) == 0xFFFF)
        {
            plrx = x;
            plry = y;
        }
        else if ((pmap[uint(y + dy)][uint(x)] & 0xFFFF) == 0xFFFF)
            plry = y;
        else if ((pmap[uint(y)][uint(x + dx)] & 0xFFFF) == 0xFFFF)
            plrx = x;
    }
    // keys.rfskey();
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
