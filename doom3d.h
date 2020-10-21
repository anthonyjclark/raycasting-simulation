//---------------------------------------------------------------------------
//--- Doom 3D engine ver: 1.000 --------------------------------------
//---------------------------------------------------------------------------
#ifndef _Doom3D_h
#define _Doom3D_h
//---------------------------------------------------------------------------
#include <math.h>
#include <jpeg.hpp>
#include "performance.h"
#include "OpenGLrep4d_double.h"
//---------------------------------------------------------------------------
const DWORD _Doom3D_cell_size = 10;  // 2D map cell size
const DWORD _Doom3D_wall_size = 100; // full height of wall in map
#define _Doom3D_filter_txr
//---------------------------------------------------------------------------
class Doom3D
{
public:
    DWORD mxs, mys, **pmap; // 2D map   // txr + height<<16
    DWORD sxs, sys, **pscr; // pseudo 3D screen
    Graphics::TBitmap *scr;
    DWORD txs, tys, **ptxr, tn;    // 2D textures
    Graphics::TBitmap *txr, *txr2; // textures, texture mipmaps resolution: /2 and /4
    double plrx, plry, plrz, plra; // player position [x,y,z,angle]
    double view_ang;               // [rad] view angle
    double focus;                  // [cells] view focal length
    struct _ray
    {
        double x, y, l; // hit or end of map position
        DWORD hit;      // map cell of hit or 0xFFFFFFFF
        char typ;       // H/V
        _ray(){};
        _ray(_ray &amp; a) { *this = a; }
        ~_ray(){};
        _ray *operator=(const _ray *a)
        {
            *this = *a;
            return this;
        }
        //_ray* operator = (const _ray &amp;a) { ..copy... return this; }
    };
    _ray *ray; // ray[sxs]

    keytab keys;
    DWORD txr_sel;
    DWORD cell_h;

    Doom3D();
    Doom3D(Doom3D &amp; a) { *this = a; }
    ~Doom3D();
    Doom3D *operator=(const Doom3D *a)
    {
        *this = *a;
        return this;
    }
    //Doom3D* operator = (const Doom3D &amp;a) { ..copy... return this; }

    void map_resize(DWORD xs, DWORD ys); // change map resolution
    void map_height(DWORD height);       // set height for whole map to convert maps from Wolfenstein3D demo
    void map_clear();                    // clear whole map
    void map_save(AnsiString name);
    void map_load(AnsiString name);
    void scr_resize(DWORD xs, DWORD ys);
    void txr_load(AnsiString name);

    void draw();
    void update(double dt);
    void mouse(double x, double y, TShiftState sh)
    {
        x = floor(x / _Doom3D_cell_size);
        if (x >= mxs)
            x = mxs - 1;
        if (x < 0)
            x = 0;
        y = floor(y / _Doom3D_cell_size);
        if (y >= mys)
            y = mys - 1;
        if (y < 0)
            y = 0;
        DWORD xx = x, yy = y;
        keys.setm(x, y, sh);
        if (keys.Shift.Contains(ssLeft))
            pmap[yy][xx] = (txr_sel) | (cell_h << 16);
        if (keys.Shift.Contains(ssRight))
            pmap[yy][xx] = 0xFFFFFFFF;
        keys.rfsmouse();
    }
    void wheel(int delta, TShiftState sh)
    {
        if (sh.Contains(ssShift))
        {
            if (delta < 0)
            {
                cell_h -= 10;
                if (cell_h < 10)
                    cell_h = 10;
            }
            if (delta > 0)
            {
                cell_h += 10;
                if (cell_h > _Doom3D_wall_size)
                    cell_h = _Doom3D_wall_size;
            }
        }
        else
        {
            if (delta < 0)
            {
                txr_sel--;
                if (txr_sel == 0xFFFFFFFF)
                    txr_sel = tn - 1;
            }
            if (delta > 0)
            {
                txr_sel++;
                if (txr_sel == tn)
                    txr_sel = 0;
            }
        }
    }
};
//---------------------------------------------------------------------------
Doom3D::Doom3D()
{
    mxs = 0;
    mys = 0;
    pmap = NULL;
    sxs = 0;
    sys = 0;
    scr = new Graphics::TBitmap;
    pscr = NULL;
    ray = NULL;
    txs = 0;
    tys = 0;
    txr = new Graphics::TBitmap;
    ptxr = NULL;
    tn = 0;
    txr2 = new Graphics::TBitmap;
    plrx = 0.0;
    plry = 0.0;
    plrz = 0.0;
    plra = 0.0;
    view_ang = 60.0 * deg;
    focus = 0.25;
    txr_sel = 0;
    cell_h = _Doom3D_wall_size;

    txr_load("textures128x128.jpg");
    map_resize(16, 16);
    map_load("Doom3D_map.dat");
}
//---------------------------------------------------------------------------
Doom3D::~Doom3D()
{
    DWORD y;
    map_save("Doom3D_map.dat");
    if (pmap)
    {
        for (y = 0; y < mys; y++)
            delete[] pmap[y];
        delete[] pmap;
        pmap = NULL;
    }
    if (ray)
        delete[] ray;
    ray = NULL;
    if (pscr)
    {
        delete[] pscr;
        pscr = NULL;
    }
    if (scr)
        delete scr;
    scr = NULL;
    if (ptxr)
    {
        delete[] ptxr;
        ptxr = NULL;
    }
    if (txr)
        delete txr;
    txr = NULL;
    if (txr2)
        delete txr2;
    txr2 = NULL;
}
//---------------------------------------------------------------------------
void Doom3D::map_resize(DWORD xs, DWORD ys)
{
    DWORD y;
    if (pmap)
    {
        for (y = 0; y < mys; y++)
            delete[] pmap[y];
        delete[] pmap;
        pmap = NULL;
    }
    mys = ys;
    mxs = xs;
    pmap = new DWORD *[mys];
    for (y = 0; y < mys; y++)
        pmap[y] = new DWORD[mxs];
    map_clear();
    plrx = (mxs - 1) * 0.5;
    plry = (mys - 1) * 0.5;
    plrz = 0.0;
    plra = 0.0 * deg;
}
//---------------------------------------------------------------------------
void Doom3D::map_height(DWORD h)
{
    DWORD x, y, c;
    for (y = 0; y < mys; y++)
        for (x = 0; x < mxs; x++)
        {
            c = pmap[y][x];
            c &amp;
            = 0xFFFF;
            c |= h << 16;
            pmap[y][x] = c;
        }
}
//---------------------------------------------------------------------------
void Doom3D::map_clear()
{
    DWORD x, y, c;
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
void Doom3D::map_save(AnsiString name)
{
    int hnd = FileCreate(name);
    if (hnd < 0)
        return;
    DWORD y;
    y = ' PAM';
    FileWrite(hnd, &amp; y, 4);   // id
    FileWrite(hnd, &amp; mxs, 4); // x resolution
    FileWrite(hnd, &amp; mys, 4); // y resolution
    for (y = 0; y < mys; y++)     // map
        FileWrite(hnd, pmap[y], mxs << 2);
    y = ' RLP';
    FileWrite(hnd, &amp; y, 4); // id
    FileWrite(hnd, &amp; plrx, 8);
    FileWrite(hnd, &amp; plry, 8);
    FileWrite(hnd, &amp; plrz, 8);
    FileWrite(hnd, &amp; plra, 8);
    FileClose(hnd);
}
//---------------------------------------------------------------------------
void Doom3D::map_load(AnsiString name)
{
    int hnd = FileOpen(name, fmOpenRead);
    if (hnd < 0)
        return;
    DWORD x, y;
    y = ' PAM';
    FileRead(hnd, &amp; x, 4); // id
    if (x == y)
    {
        FileRead(hnd, &amp; x, 4); // x resolution
        FileRead(hnd, &amp; y, 4); // y resolution
        map_resize(x, y);
        for (y = 0; y < mys; y++) // map
            FileRead(hnd, pmap[y], mxs << 2);
    }
    y = ' RLP';
    FileRead(hnd, &amp; x, 4); // id
    if (x == y)
    {
        FileRead(hnd, &amp; plrx, 8);
        FileRead(hnd, &amp; plry, 8);
        FileRead(hnd, &amp; plrz, 8);
        FileRead(hnd, &amp; plra, 8);
    }
    FileClose(hnd);
}
//---------------------------------------------------------------------------
void Doom3D::scr_resize(DWORD xs, DWORD ys)
{
    scr->HandleType = bmDIB;
    scr->PixelFormat = pf32bit;
    scr->SetSize(xs, ys);
    sxs = scr->Width;
    sys = scr->Height;
    delete[] pscr;
    pscr = new DWORD *[sys];
    for (DWORD y = 0; y < sys; y++)
        pscr[y] = (DWORD *)scr->ScanLine[y];
    if (ray)
        delete[] ray;
    ray = new _ray[sxs];
}
//---------------------------------------------------------------------------
void Doom3D::txr_load(AnsiString name)
{
    AnsiString ext = ExtractFileExt(name).LowerCase();
    for (;;)
    {
        if (ext == ".bmp")
        {
            txr->LoadFromFile(name);
            break;
        }
        if (ext == ".jpg")
        {
            TJPEGImage *jpg = new TJPEGImage;
            if (jpg == NULL)
                return;
            jpg->LoadFromFile(name);
            txr->Assign(jpg);
            delete jpg;
            break;
        }
        return;
    }
    DWORD y = tys;
    txr->HandleType = bmDIB;
    txr->PixelFormat = pf32bit;
    txs = txr->Width;
    tys = txr->Height;
    // mip map
    txr2->SetSize(txs >> 1, (tys >> 1) + (tys >> 2));
    txr2->Canvas->StretchDraw(TRect(0, 0, txs >> 1, tys >> 1), txr);
    txr2->Canvas->StretchDraw(TRect(0, tys >> 1, txs >> 2, (tys >> 1) + (tys >> 2)), txr);
    tn = txs / tys;
    txs = tys;
    delete[] ptxr;
    ptxr = new DWORD *[tys];
    for (y = 0; y < tys; y++)
        ptxr[y] = (DWORD *)txr->ScanLine[y];
}
//---------------------------------------------------------------------------
void Doom3D::draw()
{
    // total time measurement
    tbeg();
    double tperf0 = performance_tms;

    AnsiString tcls, tray, tmap, ttotal;
    double a, a0, da, dx, dy, l, mx, my;
    DWORD x, y, xs2, ys2, c, m;
    double xx0, yy0, dx0, dy0, ll0;
    DWORD c0, d0;
    double xx1, yy1, dx1, dy1, ll1;
    DWORD c1, d1;
    _ray *p;
    xs2 = sxs >> 1;
    ys2 = sys >> 1;

    // aspect ratio,view angle corrections
    a = 90.0 * deg - view_ang;
    double wall = double(sxs) * (1.25 + (0.288 * a) + (2.04 * a * a)); // [px]

    // floor,ceilling/sky
    tbeg();
    for (y = 0; y < ys2; y++)
        for (x = 0; x < sxs; x++)
            pscr[y][x] = 0x000080FF;
    for (; y < sys; y++)
        for (x = 0; x < sxs; x++)
            pscr[y][x] = 0x00404040;
    tend();
    tcls = tstr(1) + " cls";

    // [cast rays]
    tbeg();
    // diffuse + ambient lighting
    DWORD ch = 155.0 + fabs(100.0 * sin(plra));
    DWORD cv = 155.0 + fabs(100.0 * cos(plra));
    a0 = plra - (0.5 * view_ang);
    da = divide(view_ang, sxs - 1);
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
            dx1 = divide(1.0, tan(a));
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
            if ((c0)&amp; &amp; (ll0 < ll1))
            {
                m = DWORD(xx0 - dx0);
                if ((m >= 0.0) & amp; &amp; (m < mxs) & amp; &amp; (!_bck))
                {
                    c = pmap[DWORD(yy0)][m];
                    if ((c & amp; 0xFFFF) != 0xFFFF)
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
                    c = pmap[DWORD(yy0)][DWORD(xx0)];
                    if ((c & amp; 0xFFFF) != 0xFFFF)
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
            if ((c1)&amp; &amp; (ll0 > ll1) & amp; &amp; (!_hit))
            {
                m = DWORD(yy1 - dy1);
                if ((m >= 0.0) & amp; &amp; (m < mys) & amp; &amp; (!_bck))
                {
                    c = pmap[m][DWORD(xx1)];
                    if ((c & amp; 0xFFFF) != 0xFFFF)
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
                    c = pmap[DWORD(yy1)][DWORD(xx1)];
                    if ((c & amp; 0xFFFF) != 0xFFFF)
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
                    DWORD dd;
                    BYTE db[4];
                } cc;
                int tx, ty, sy, sy0, sy1, cnt, dsy, dty;
                p->l = sqrt(p->l) * cos(a - plra); // anti fish eye
                m = divide(wall * focus, p->l);    // projected wall half size
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
                tx += txs * (p->hit & amp; 0xFFFF);

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
                DWORD r = 0, g = 0, b = 0, n = 0;
#else
                cc.dd = ptxr[ty][tx];
                cc.db[0] = DWORD((DWORD(cc.db[0]) * c) >> 8);
                cc.db[1] = DWORD((DWORD(cc.db[1]) * c) >> 8);
                cc.db[2] = DWORD((DWORD(cc.db[2]) * c) >> 8);
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
                        if ((sy > 0) & amp; &amp; (sy < sys))
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
                            b += DWORD(cc.db[0]);
                            g += DWORD(cc.db[1]);
                            r += DWORD(cc.db[2]);
                            n += 256;
                        }
                        if ((sy > 0) & amp; &amp; (sy < sys))
                        {
                            cc.db[0] = DWORD(c * b / n);
                            b = 0;
                            cc.db[1] = DWORD(c * g / n);
                            g = 0;
                            cc.db[2] = DWORD(c * r / n);
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
                            b += DWORD(cc.db[0]);
                            g += DWORD(cc.db[1]);
                            r += DWORD(cc.db[2]);
                            n += 256;
                        }
#else
                        if ((sy > 0) & amp; &amp; (sy < sys))
                            pscr[sy][x] = cc.dd;
                        cnt -= dty;
                        while (cnt <= 0)
                        {
                            cnt += dsy;
                            ty++;
                            cc.dd = ptxr[ty][tx];
                            cc.db[0] = DWORD((DWORD(cc.db[0]) * c) >> 8);
                            cc.db[1] = DWORD((DWORD(cc.db[1]) * c) >> 8);
                            cc.db[2] = DWORD((DWORD(cc.db[2]) * c) >> 8);
                        }
#endif
                    }
                if (height0 < 0)
                    break;
            }
        }
    }
    tend();
    tray = tstr(1) + " ray";

    // [2D map]
    tbeg();
    m = _Doom3D_cell_size;
    mx = _Doom3D_cell_size;
    if ((sxs >= mxs * m) & amp; &amp; (sys >= mys * m))
    {
        for (y = 0; y < mys * m; y++) // pmap[][]
            for (x = 0; x < mxs * m; x++)
            {
                if ((pmap[y / m][x / m] & amp; 0xFFFF) != 0xFFFF)
                    c = 0x00808080;
                else
                    c = 0x00000000;
                pscr[y][x] = c;
            }
        x = double(plrx * mx); // view rays
        y = double(plry * mx);
        scr->Canvas->Pen->Color = 0x00005050;
        scr->Canvas->Pen->Mode = pmMerge;
        for (c = 0; c < sxs; c++)
        {
            scr->Canvas->MoveTo(x, y);
            scr->Canvas->LineTo(DWORD(ray[c].x * mx), DWORD(ray[c].y * mx));
        }
        scr->Canvas->Pen->Mode = pmCopy;
        c = focus * m; // player and view direction
        scr->Canvas->Pen->Color = 0x000000FF;
        scr->Canvas->Brush->Color = 0x000000FF;
        scr->Canvas->MoveTo(x, y);
        scr->Canvas->LineTo(DWORD(ray[xs2].x * mx), DWORD(ray[xs2].y * mx));
        scr->Canvas->Ellipse(x - c, y - c, x + c, y + c);
        scr->Canvas->Pen->Color = 0x00202020;
        for (y = 0; y <= mys; y++) // map grid
            for (x = 0; x <= mxs; x++)
            {
                scr->Canvas->MoveTo(0, y * m);
                scr->Canvas->LineTo(mxs * m, y * m);
                scr->Canvas->MoveTo(x * m, 0);
                scr->Canvas->LineTo(x * m, mys * m);
            }
        x = keys.mx * m; // selected cell
        y = keys.my * m;
        scr->Canvas->Pen->Color = 0x0020FFFF;
        scr->Canvas->MoveTo(x, y);
        scr->Canvas->LineTo(x + m, y);
        scr->Canvas->LineTo(x + m, y + m);
        scr->Canvas->LineTo(x, y + m);
        scr->Canvas->LineTo(x, y);
    }
    tend();
    tmap = tstr(1) + " map";

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
                scr->Canvas->CopyRect(TRect(x - s1, y + (s1 >> 1), x, s1 + (s1 >> 1)), txr2->Canvas, TRect(s1 * j, s0, s1 * j + s1, s0 + s1));
                x -= s1 + 5;
            }
            else
            {
                scr->Canvas->CopyRect(TRect(x - s0, y + s0 - s2, x, s0), txr2->Canvas, TRect(s0 * j, 0, s0 * j + s0, s2));
                x -= s0 + 5;
            }
        }
    }

    // total time measurement
    performance_tms = tperf0;
    tend();
    ttotal = tstr(1) + " total";

    x = m * mxs + m;
    c = 16;
    y = -c;
    scr->Canvas->Font->Color = clYellow;
    scr->Canvas->Brush->Style = bsClear;
    scr->Canvas->TextOutA(x, y += c, AnsiString().sprintf("player: %.2lf x %.2lf x %.2lf", plrx, plry, plrz));
    scr->Canvas->TextOutA(x, y += c, AnsiString().sprintf(" mouse: %.2lf x %.2lf", keys.mx, keys.my));
    scr->Canvas->TextOutA(x, y += c, tray);
    scr->Canvas->TextOutA(x, y += c, tcls);
    scr->Canvas->TextOutA(x, y += c, tmap);
    scr->Canvas->TextOutA(x, y += c, ttotal);
    scr->Canvas->TextOutA(x, y += c, AnsiString().sprintf("   key: %d", keys.Key));

    // aspect ratio test
    /*
    c=ys2*7/10;
    scr->Canvas->Rectangle(xs2-c,ys2-c,xs2+c,ys2+c);
*/
    // cross
    c = 4, m = 32;
    scr->Canvas->Pen->Color = clRed;
    scr->Canvas->MoveTo(xs2 - c, ys2 - m);
    scr->Canvas->LineTo(xs2 - c, ys2 - c);
    scr->Canvas->LineTo(xs2 - m, ys2 - c);
    scr->Canvas->MoveTo(xs2 + c, ys2 - m);
    scr->Canvas->LineTo(xs2 + c, ys2 - c);
    scr->Canvas->LineTo(xs2 + m, ys2 - c);
    scr->Canvas->MoveTo(xs2 - c, ys2 + m);
    scr->Canvas->LineTo(xs2 - c, ys2 + c);
    scr->Canvas->LineTo(xs2 - m, ys2 + c);
    scr->Canvas->MoveTo(xs2 + c, ys2 + m);
    scr->Canvas->LineTo(xs2 + c, ys2 + c);
    scr->Canvas->LineTo(xs2 + m, ys2 + c);

    scr->Canvas->Brush->Style = bsSolid;
}
//---------------------------------------------------------------------------
void Doom3D::update(double dt)
{
    int move = 0;
    double da = 120.0 * deg * dt;
    double dl = 5.0 * dt;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    if (keys.get(104))
    {
        plra -= da;
        if (plra < 0.0)
            plra += pi2;
    } // turn l/r
    if (keys.get(105))
    {
        plra += da;
        if (plra >= pi2)
            plra -= pi2;
    }
    if (keys.get(101))
    {
        move = 1;
        dx = +dl * cos(plra);
        dy = +dl * sin(plra);
    } // move f/b
    if (keys.get(98))
    {
        move = 1;
        dx = -dl * cos(plra);
        dy = -dl * sin(plra);
    }
    if (keys.get(102))
    {
        move = 1;
        dx = dl * cos(plra - 90 * deg);
        dy = dl * sin(plra - 90 * deg);
    } // strafe l/r
    if (keys.get(99))
    {
        move = 1;
        dx = dl * cos(plra + 90 * deg);
        dy = dl * sin(plra + 90 * deg);
    }
    if (keys.get(100))
    {
        move = 1;
        dz = +dl;
    } // strafe u/d
    if (keys.get(97))
    {
        move = 1;
        dz = -dl;
    }
    if (move) // update/test plr position
    {
        double x, y, z, mx, my;
        x = plrx + dx;
        mx = mxs - focus;
        y = plry + dy;
        my = mys - focus;
        z = plrz + dz;
        if ((z >= 0.0) & amp; &amp; (z <= _Doom3D_wall_size))
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
        dx *= divide(focus, dl);
        dy *= divide(focus, dl);
        if ((pmap[DWORD(y + dy)][DWORD(x + dx)] & amp; 0xFFFF) == 0xFFFF)
        {
            plrx = x;
            plry = y;
        }
        else if ((pmap[DWORD(y + dy)][DWORD(x)] & amp; 0xFFFF) == 0xFFFF)
            plry = y;
        else if ((pmap[DWORD(y)][DWORD(x + dx)] & amp; 0xFFFF) == 0xFFFF)
            plrx = x;
    }
    keys.rfskey();
}
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
#endif
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
