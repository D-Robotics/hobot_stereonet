#ifndef __gc_vdk_types_h_
#define __gc_vdk_types_h_

#ifdef _WIN32
#pragma warning(disable:4127)    /* Conditional expression is constant. */
#pragma warning(disable:4100)    /* Unreferenced formal parameter. */
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <EGL/egl.h>


/*******************************************************************************
** vdkPrivate. *****************************************************************
*/

typedef struct _vdkPrivate *    vdkPrivate;

/*******************************************************************************
** Display. ********************************************************************
*/
typedef EGLNativeDisplayType    vdkDisplay;

/*******************************************************************************
** Window. *********************************************************************
*/
typedef EGLNativeWindowType     vdkWindow;


/*******************************************************************************
** Pixmap. *********************************************************************
*/
typedef EGLNativePixmapType     vdkPixmap;


/*******************************************************************************
** ClientBuffer. ***************************************************************
*/
typedef void *                  vdkClientBuffer;

/*******************************************************************************
** Events. *********************************************************************
*/

/* Scancodes for keyboard. */
typedef enum _vdkKeys
{
    VDK_UNKNOWN = -1,

    VDK_BACKSPACE = 0x08,
    VDK_TAB,
    VDK_ENTER = 0x0D,
    VDK_ESCAPE = 0x1B,

    VDK_SPACE = 0x20,
    VDK_SINGLEQUOTE = 0x27,
    VDK_PAD_ASTERISK = 0x2A,
    VDK_COMMA = 0x2C,
    VDK_HYPHEN,
    VDK_PERIOD,
    VDK_SLASH,
    VDK_0,
    VDK_1,
    VDK_2,
    VDK_3,
    VDK_4,
    VDK_5,
    VDK_6,
    VDK_7,
    VDK_8,
    VDK_9,
    VDK_SEMICOLON = 0x3B,
    VDK_EQUAL = 0x3D,
    VDK_A = 0x41,
    VDK_B,
    VDK_C,
    VDK_D,
    VDK_E,
    VDK_F,
    VDK_G,
    VDK_H,
    VDK_I,
    VDK_J,
    VDK_K,
    VDK_L,
    VDK_M,
    VDK_N,
    VDK_O,
    VDK_P,
    VDK_Q,
    VDK_R,
    VDK_S,
    VDK_T,
    VDK_U,
    VDK_V,
    VDK_W,
    VDK_X,
    VDK_Y,
    VDK_Z,
    VDK_LBRACKET,
    VDK_BACKSLASH,
    VDK_RBRACKET,
    VDK_BACKQUOTE = 0x60,

    VDK_F1 = 0x80,
    VDK_F2,
    VDK_F3,
    VDK_F4,
    VDK_F5,
    VDK_F6,
    VDK_F7,
    VDK_F8,
    VDK_F9,
    VDK_F10,
    VDK_F11,
    VDK_F12,

    VDK_LCTRL,
    VDK_RCTRL,
    VDK_LSHIFT,
    VDK_RSHIFT,
    VDK_LALT,
    VDK_RALT,
    VDK_CAPSLOCK,
    VDK_NUMLOCK,
    VDK_SCROLLLOCK,
    VDK_PAD_0,
    VDK_PAD_1,
    VDK_PAD_2,
    VDK_PAD_3,
    VDK_PAD_4,
    VDK_PAD_5,
    VDK_PAD_6,
    VDK_PAD_7,
    VDK_PAD_8,
    VDK_PAD_9,
    VDK_PAD_HYPHEN,
    VDK_PAD_PLUS,
    VDK_PAD_SLASH,
    VDK_PAD_PERIOD,
    VDK_PAD_ENTER,
    VDK_SYSRQ,
    VDK_PRNTSCRN,
    VDK_BREAK,
    VDK_UP,
    VDK_LEFT,
    VDK_RIGHT,
    VDK_DOWN,
    VDK_HOME,
    VDK_END,
    VDK_PGUP,
    VDK_PGDN,
    VDK_INSERT,
    VDK_DELETE,
    VDK_LWINDOW,
    VDK_RWINDOW,
    VDK_MENU,
    VDK_POWER,
    VDK_SLEEP,
    VDK_WAKE,
    VDK_MUTE,
    VDK_VOLUMEDOWN,
    VDK_VOLUMEUP,
    VDK_BACK,
    VDK_FORWARD,
}
vdkKeys;

typedef enum _vdkEventType
{
    /* Keyboard event. */
    VDK_KEYBOARD,

    /* Mouse move event. */
    VDK_POINTER,

    /* Mouse button event. */
    VDK_BUTTON,

    /* Application close event. */
    VDK_CLOSE,

    /* Application window has been updated. */
    VDK_WINDOW_UPDATE
}
vdkEventType;


/* Event structure. */
typedef struct _vdkEvent
{
    /* Event type. */
    vdkEventType type;

    /* Event data union. */
    union _vdkEventData
    {
        /* Event data for keyboard. */
        struct _vdkKeyboard
        {
            /* Scancode. */
            vdkKeys    scancode;

            /* ASCII characte of the key pressed. */
            char    key;

            /* Flag whether the key was pressed (1) or released (0). */
            char    pressed;
        }
        keyboard;

        /* Event data for pointer. */
        struct _vdkPointer
        {
            /* Current pointer coordinate. */
            int        x;
            int        y;
        }
        pointer;

        /* Event data for mouse buttons. */
        struct _vdkButton
        {
            /* Left button state. */
            int        left;

            /* Middle button state. */
            int        middle;

            /* Right button state. */
            int        right;

            /* Current pointer coordinate. */
            int        x;
            int        y;
        }
        button;
    }
    data;
}
vdkEvent;

#ifdef __cplusplus
}
#endif

#endif /* __gc_vdk_types_h_ */
