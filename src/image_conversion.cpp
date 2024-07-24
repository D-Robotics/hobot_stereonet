//
// Created by zhy on 7/22/24.
//

#include <arm_neon.h>
#include <cstdint>
#include "image_conversion.h"

void image_conversion::nv12_to_bgr24_neon(uint8_t* nv12, uint8_t* bgr24, int width, int height) {
  const uint8_t* yptr = nv12;
  const uint8_t* uvptr = nv12 + width * height;
  uint8x8_t _v128 = vdup_n_u8(128);
  int8x8_t _v127= vdup_n_s8(127);
  uint8x8_t _v16 = vdup_n_u8(16);
  uint8x8_t _v75 = vdup_n_u8(75);
  uint8x8_t _vu64 = vdup_n_u8(64);
  int8x8_t _v52 = vdup_n_s8(52);
  int8x8_t _v25 = vdup_n_s8(25);
  int8x8_t _v102 = vdup_n_s8(102);
  int16x8_t _v64 = vdupq_n_s16(64);

  for (int y = 0; y < height; y += 2)
  {
    const uint8_t* yptr0 = yptr;
    const uint8_t* yptr1 = yptr + width;
    unsigned char* rgb0 = bgr24;
    unsigned char* rgb1 = bgr24 + width * 3;
    int nn = width >> 3;

    for (; nn > 0; nn--)
    {
      int16x8_t _yy0 = vreinterpretq_s16_u16(vmull_u8(vqsub_u8(vld1_u8(yptr0), _v16), _v75));
      int16x8_t _yy1 = vreinterpretq_s16_u16(vmull_u8(vqsub_u8(vld1_u8(yptr1), _v16), _v75));
//      int16x8_t _yy0 = vreinterpretq_s16_u16(vmull_u8(vld1_u8(yptr0), _v75));
//      int16x8_t _yy1 = vreinterpretq_s16_u16(vmull_u8(vld1_u8(yptr1), _v75));
      int8x8_t _uuvv = vreinterpret_s8_u8(vsub_u8(vld1_u8(uvptr), _v128));
      int8x8x2_t _uuuuvvvv = vtrn_s8(_uuvv, _uuvv);
      int8x8_t _uu = _uuuuvvvv.val[0];
      int8x8_t _vv = _uuuuvvvv.val[1];

      int16x8_t _r0 = vmlal_s8(_yy0, _vv, _v102);
      int16x8_t _g0 = vmlsl_s8(_yy0, _vv, _v52);
      _g0 = vmlsl_s8(_g0, _uu, _v25);
      int16x8_t _b0 = vmlal_s8(_yy0, _uu, _v127);


      int16x8_t _r1 = vmlal_s8(_yy1, _vv, _v102);
      int16x8_t _g1 = vmlsl_s8(_yy1, _vv, _v52);
      _g1 = vmlsl_s8(_g1, _uu, _v25);
      int16x8_t _b1 = vmlal_s8(_yy1, _uu, _v127);

      uint8x8x3_t _rgb0;
      _rgb0.val[2] = vqshrun_n_s16(vaddq_s16(_r0, _v64), 6);
      _rgb0.val[1] = vqshrun_n_s16(vaddq_s16(_g0, _v64), 6);
      _rgb0.val[0] = vqshrun_n_s16(vaddq_s16(_b0, _v64), 6);

      uint8x8x3_t _rgb1;
      _rgb1.val[2] = vqshrun_n_s16(vaddq_s16(_r1, _v64), 6);
      _rgb1.val[1] = vqshrun_n_s16(vaddq_s16(_g1, _v64), 6);
      _rgb1.val[0] = vqshrun_n_s16(vaddq_s16(_b1, _v64), 6);

      vst3_u8(rgb0, _rgb0);
      vst3_u8(rgb1, _rgb1);

      yptr0 += 8;
      yptr1 += 8;
      uvptr += 8;
      rgb0 += 24;
      rgb1 += 24;
    }
    yptr += 2 * width;
    bgr24 += 2 * 3 * width;
  }
}

void image_conversion::bgr24_to_nv12_neon(uint8_t* bgr24, uint8_t* nv12, int width, int height) {
  int frameSize = width * height;
  int yIndex = 0;
  int uvIndex = frameSize;
  const uint16x8_t u16_rounding = vdupq_n_u16(128);
  const int16x8_t s16_rounding = vdupq_n_s16(128);
  const int8x8_t s8_rounding = vdup_n_s8(128);
  const uint8x16_t offset = vdupq_n_u8(16);
  const uint16x8_t mask = vdupq_n_u16(255);

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width >> 4; i++) {
      // Load rgb
      uint8x16x3_t pixel_rgb;
      pixel_rgb = vld3q_u8(bgr24);
      bgr24 += 48;

      uint8x8x2_t uint8_r;
      uint8x8x2_t uint8_g;
      uint8x8x2_t uint8_b;
      uint8_r.val[0] = vget_low_u8(pixel_rgb.val[2]);
      uint8_r.val[1] = vget_high_u8(pixel_rgb.val[2]);
      uint8_g.val[0] = vget_low_u8(pixel_rgb.val[1]);
      uint8_g.val[1] = vget_high_u8(pixel_rgb.val[1]);
      uint8_b.val[0] = vget_low_u8(pixel_rgb.val[0]);
      uint8_b.val[1] = vget_high_u8(pixel_rgb.val[0]);

      uint16x8x2_t uint16_y;
      uint8x8_t scalar = vdup_n_u8(66);
      uint8x16_t y;

      uint16_y.val[0] = vmull_u8(uint8_r.val[0], scalar);
      uint16_y.val[1] = vmull_u8(uint8_r.val[1], scalar);
      scalar = vdup_n_u8(129);
      uint16_y.val[0] = vmlal_u8(uint16_y.val[0], uint8_g.val[0], scalar);
      uint16_y.val[1] = vmlal_u8(uint16_y.val[1], uint8_g.val[1], scalar);
      scalar = vdup_n_u8(25);
      uint16_y.val[0] = vmlal_u8(uint16_y.val[0], uint8_b.val[0], scalar);
      uint16_y.val[1] = vmlal_u8(uint16_y.val[1], uint8_b.val[1], scalar);

      uint16_y.val[0] = vaddq_u16(uint16_y.val[0], u16_rounding);
      uint16_y.val[1] = vaddq_u16(uint16_y.val[1], u16_rounding);

      y = vcombine_u8(vqshrn_n_u16(uint16_y.val[0], 8), vqshrn_n_u16(uint16_y.val[1], 8));
      y = vaddq_u8(y, offset);

      vst1q_u8(nv12 + yIndex, y);
      yIndex += 16;

      // Compute u and v in the even row
      if (j % 2 == 0) {
        int16x8_t u_scalar = vdupq_n_s16(-38);
        int16x8_t v_scalar = vdupq_n_s16(112);

        int16x8_t r = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[2]), mask));
        int16x8_t g = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[1]), mask));
        int16x8_t b = vreinterpretq_s16_u16(vandq_u16(vreinterpretq_u16_u8(pixel_rgb.val[0]), mask));

        int16x8_t u;
        int16x8_t v;
        uint8x8x2_t uv;

        u = vmulq_s16(r, u_scalar);
        v = vmulq_s16(r, v_scalar);

        u_scalar = vdupq_n_s16(-74);
        v_scalar = vdupq_n_s16(-94);
        u = vmlaq_s16(u, g, u_scalar);
        v = vmlaq_s16(v, g, v_scalar);

        u_scalar = vdupq_n_s16(112);
        v_scalar = vdupq_n_s16(-18);
        u = vmlaq_s16(u, b, u_scalar);
        v = vmlaq_s16(v, b, v_scalar);

        u = vaddq_s16(u, s16_rounding);
        v = vaddq_s16(v, s16_rounding);

        uv.val[0] = vreinterpret_u8_s8(vadd_s8(vqshrn_n_s16(u, 8), s8_rounding));
        uv.val[1] = vreinterpret_u8_s8(vadd_s8(vqshrn_n_s16(v, 8), s8_rounding));

        vst2_u8(nv12 + uvIndex, uv);

        uvIndex += 16;
      }
    }
  }
}
