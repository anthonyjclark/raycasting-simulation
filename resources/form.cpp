//$$---- Form CPP ----
//---------------------------------------------------------------------------
#include <vcl.h>
#pragma hdrstop
#include "win_main.h"
#include "Doom3D.h"
//---------------------------------------------------------------------------
#pragma package(smart_init)
#pragma resource "*.dfm"
TMain *Main;
Doom3D game;
//---------------------------------------------------------------------------
void TMain::draw()
{
    game.draw();
    Canvas->Draw(0, 0, game.scr);
}
//---------------------------------------------------------------------------
__fastcall TMain::TMain(TComponent *Owner) : TForm(Owner)
{
}
//---------------------------------------------------------------------------
void __fastcall TMain::FormResize(TObject *Sender)
{
    game.scr_resize(ClientWidth, ClientHeight);
}
//---------------------------------------------------------------------------
void __fastcall TMain::tim_redrawTimer(TObject *Sender)
{
    game.update(tim_redraw->Interval * 0.001);
    draw();
}
//---------------------------------------------------------------------------
void __fastcall TMain::FormKeyDown(TObject *Sender, WORD &amp; Key, TShiftState Shift) { game.keys.set(Key, Shift); }
void __fastcall TMain::FormKeyUp(TObject *Sender, WORD &amp; Key, TShiftState Shift) { game.keys.rst(Key, Shift); }
void __fastcall TMain::FormActivate(TObject *Sender) { game.keys.reset_keys(); }
//---------------------------------------------------------------------------
void __fastcall TMain::FormMouseMove(TObject *Sender, TShiftState Shift, int X, int Y) { game.mouse(X, Y, Shift); }
void __fastcall TMain::FormMouseDown(TObject *Sender, TMouseButton Button, TShiftState Shift, int X, int Y) { game.mouse(X, Y, Shift); }
void __fastcall TMain::FormMouseUp(TObject *Sender, TMouseButton Button, TShiftState Shift, int X, int Y) { game.mouse(X, Y, Shift); }
void __fastcall TMain::FormMouseWheel(TObject *Sender, TShiftState Shift, int WheelDelta, TPoint &amp; MousePos, bool &amp; Handled)
{
    game.wheel(WheelDelta, Shift);
    Handled = true;
}
//---------------------------------------------------------------------------
