/*
 * Vivante specific definitions and declarations for EGL library.
 */

#ifndef __eglvivante_h_
#define __eglvivante_h_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum _platEventType
{
    /* Keyboard event. */
    EVENT_KEYBOARD,

    /* Mouse move event. */
    EVENT_POINTER,

    /* Mouse button event. */
    EVENT_BUTTON,

    /* Application close event. */
    EVENT_CLOSE,

    /* Application window has been updated. */
    EVENT_WINDOW_UPDATE
}
platEventType;

/* Scancodes for keyboard. */
typedef enum _platKeyCode
{
    KEYCODE_UNKNOWN = -1,

    KEYCODE_BACKSPACE = 0x08,
    KEYCODE_TAB,
    KEYCODE_ENTER = 0x0D,
    KEYCODE_ESCAPE = 0x1B,

    KEYCODE_SPACE = 0x20,
    KEYCODE_SINGLEQUOTE = 0x27,
    KEYCODE_PAD_ASTERISK = 0x2A,
    KEYCODE_COMMA = 0x2C,
    KEYCODE_HYPHEN,
    KEYCODE_PERIOD,
    KEYCODE_SLASH,
    KEYCODE_0,
    KEYCODE_1,
    KEYCODE_2,
    KEYCODE_3,
    KEYCODE_4,
    KEYCODE_5,
    KEYCODE_6,
    KEYCODE_7,
    KEYCODE_8,
    KEYCODE_9,
    KEYCODE_SEMICOLON = 0x3B,
    KEYCODE_EQUAL = 0x3D,
    KEYCODE_A = 0x41,
    KEYCODE_B,
    KEYCODE_C,
    KEYCODE_D,
    KEYCODE_E,
    KEYCODE_F,
    KEYCODE_G,
    KEYCODE_H,
    KEYCODE_I,
    KEYCODE_J,
    KEYCODE_K,
    KEYCODE_L,
    KEYCODE_M,
    KEYCODE_N,
    KEYCODE_O,
    KEYCODE_P,
    KEYCODE_Q,
    KEYCODE_R,
    KEYCODE_S,
    KEYCODE_T,
    KEYCODE_U,
    KEYCODE_V,
    KEYCODE_W,
    KEYCODE_X,
    KEYCODE_Y,
    KEYCODE_Z,
    KEYCODE_LBRACKET,
    KEYCODE_BACKSLASH,
    KEYCODE_RBRACKET,
    KEYCODE_BACKQUOTE = 0x60,

    KEYCODE_F1 = 0x80,
    KEYCODE_F2,
    KEYCODE_F3,
    KEYCODE_F4,
    KEYCODE_F5,
    KEYCODE_F6,
    KEYCODE_F7,
    KEYCODE_F8,
    KEYCODE_F9,
    KEYCODE_F10,
    KEYCODE_F11,
    KEYCODE_F12,

    KEYCODE_LCTRL,
    KEYCODE_RCTRL,
    KEYCODE_LSHIFT,
    KEYCODE_RSHIFT,
    KEYCODE_LALT,
    KEYCODE_RALT,
    KEYCODE_CAPSLOCK,
    KEYCODE_NUMLOCK,
    KEYCODE_SCROLLLOCK,
    KEYCODE_PAD_0,
    KEYCODE_PAD_1,
    KEYCODE_PAD_2,
    KEYCODE_PAD_3,
    KEYCODE_PAD_4,
    KEYCODE_PAD_5,
    KEYCODE_PAD_6,
    KEYCODE_PAD_7,
    KEYCODE_PAD_8,
    KEYCODE_PAD_9,
    KEYCODE_PAD_HYPHEN,
    KEYCODE_PAD_PLUS,
    KEYCODE_PAD_SLASH,
    KEYCODE_PAD_PERIOD,
    KEYCODE_PAD_ENTER,
    KEYCODE_SYSRQ,
    KEYCODE_PRNTSCRN,
    KEYCODE_BREAK,
    KEYCODE_UP,
    KEYCODE_LEFT,
    KEYCODE_RIGHT,
    KEYCODE_DOWN,
    KEYCODE_HOME,
    KEYCODE_END,
    KEYCODE_PGUP,
    KEYCODE_PGDN,
    KEYCODE_INSERT,
    KEYCODE_DELETE,
    KEYCODE_LWINDOW,
    KEYCODE_RWINDOW,
    KEYCODE_MENU,
    KEYCODE_POWER,
    KEYCODE_SLEEP,
    KEYCODE_WAKE
}
platKeyCode;

/* Event structure. */
typedef struct _platEvent
{
    /* Event type. */
    platEventType type;

    /* Event data union. */
    union
    {
        /* Event data for keyboard. */
        struct _platKeyboard
        {
            /* Scancode. */
            platKeyCode scancode;

            /* ASCII characte of the key pressed. */
            char    key;

            /* Flag whether the key was pressed (1) or released (0). */
            char    pressed;
        }
        keyboard;

        /* Event data for pointer. */
        struct _platPointer
        {
            /* Current pointer coordinate. */
            int     x;
            int     y;
        }
        pointer;

        /* Event data for mouse buttons. */
        struct _platButton
        {
            /* Left button state. */
            int     left;

            /* Middle button state. */
            int     middle;

            /* Right button state. */
            int     right;

            /* Current pointer coordinate. */
            int     x;
            int     y;
        }
        button;
    }
    data;
}
platEvent;

#if (defined(__linux__) && defined(EGL_API_FB) && !defined(__APPLE__)) \
    || (defined(__VXWORKS__))

void *
fbGetDisplay(
    void *context
    );

void *
fbGetDisplayByIndex(
    int DisplayIndex
    );

void
fbGetDisplayGeometry(
    void * Display,
    int * Width,
    int * Height
    );

void
fbGetDisplayInfo(
    void * Display,
    int * Width,
    int * Height,
    unsigned long * Physical,
    int * Stride,
    int * BitsPerPixel
    );

void
fbDestroyDisplay(
    void * Display
    );

void *
fbCreateWindow(
    void * Display,
    int X,
    int Y,
    int Width,
    int Height
    );

void
fbGetWindowGeometry(
    void * Window,
    int * X,
    int * Y,
    int * Width,
    int * Height
    );

void
fbGetWindowInfo(
    void * Window,
    int * X,
    int * Y,
    int * Width,
    int * Height,
    int * BitsPerPixel,
    unsigned int * Offset
    );

void
fbDestroyWindow(
    void * Window
    );

void *
fbCreatePixmap(
    void * Display,
    int Width,
    int Height
    );

void *
fbCreatePixmapWithBpp(
    void * Display,
    int Width,
    int Height,
    int BitsPerPixel
    );

void
fbGetPixmapGeometry(
    void * Pixmap,
    int * Width,
    int * Height
    );

void
fbGetPixmapInfo(
    void * Pixmap,
    int * Width,
    int * Height,
    int * BitsPerPixel,
    int * Stride,
    void ** Bits
    );

void
fbDestroyPixmap(
    void * Pixmap
    );

#endif

#if defined(__linux__) && defined(EGL_API_DFB) && !defined(__APPLE__)

void *
dfbGetDisplay(
    void *context
    );

void
dfbDestroyDisplay(
    void * Display
    );

void *
dfbCreateWindow(
    void * Display,
    int X,
    int Y,
    int Width,
    int Height
    );

void
dfbShowWindow(
    void * Display,
    void * Window
    );

void
dfbHideWindow(
    void * Display,
    void * Window
    );

int
dfbGetEvent(
    void * Display,
    void * Window,
    platEvent * Event
    );

int
dfbGetWindowInfo(
    void * Window,
    int * X,
    int * Y,
    int * Width,
    int * Height,
    int * BitsPerPixel,
    unsigned int * Offset
    );

void
dfbDestroyWindow(
    void * Window
    );

void *
dfbCreatePixmap(
    void * Display,
    int Width,
    int Height
    );

void *
dfbCreatePixmapWithBpp(
    void * Display,
    int Width,
    int Height,
    int BitsPerPixel
    );

void
dfbGetPixmapInfo(
    void * Pixmap,
    int * Width,
    int * Height,
    int * BitsPerPixel,
    int * Stride,
    void* * Bits
    );

void
dfbDestroyPixmap(
    void * Pixmap
    );

#endif

#if defined(EGL_API_WL)
/* Wayland */

/**
 * wl_egl_pixmap - Vivante specific pixmap implementation on wayland.
 *
 * wayland actually does not define pixmap.
 */
struct wl_egl_pixmap;

/**
 * wl_egl_pixmap_create - Create wayland pixmap.
 * @format uses 'enum wl_shm_format'
 */
struct wl_egl_pixmap *wl_egl_pixmap_create(int width, int height, int format);

/**
 * wl_egl_pixmap_destroy - Destroy wayland pixmap.
 */
void wl_egl_pixmap_destroy(struct wl_egl_pixmap *pixmap);

/**
 * wl_egl_pixmap_get_stride - Obtain stride in bytes of the pixmap
 *
 * Stride is invalid before eglCreatePixmapSurface.
 */
void wl_egl_pixmap_get_stride(struct wl_egl_pixmap *pixmap, int *stride);

/**
 * wl_egl_pixmap_lock - Lock pixmap memory for client read or write.
 * @pixels pointer to hold client virtual address.
 *
 * Pixel memory is invalid before eglCreatePixmapSurface.
 * Multiple lock will fail, and pixels sets to NULL.
 */
void wl_egl_pixmap_lock(struct wl_egl_pixmap *pixmap, void **pixels);

/**
 * wl_egl_pixmap_unlock - Unlock pixmap.
 *
 * Pixel memory is invalid before eglCreatePixmapSurface.
 */
void wl_egl_pixmap_unlock(struct wl_egl_pixmap *pixmap);

#endif


#ifdef __cplusplus
}
#endif

#endif /* __eglvivante_h_ */
