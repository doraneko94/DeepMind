#include "DxLib/DxLib.h"
// #define DDX_GCC_COMPILE
// #define DDX_NON_INLINE_ASM

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    if (DxLib_Init() == -1) {
        return -1;
    }

    DrawPixel(320, 240, GetColor(255,255,255));

    WaitKey();

    DxLib_End();

    return 0;
}