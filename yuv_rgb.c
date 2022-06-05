// Copyright 2016 Adrien Descamps
// Distributed under BSD 3-Clause License

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "yuv_rgb.h"

#include <emmintrin.h>
#include <immintrin.h>

#ifdef _MSC_VER
// MSVC does not have __SSE2__ macro
  #if (defined(_M_AMD64) || defined(_M_X64) || (_M_IX86_FP == 2))
    #define _YUVRGB_SSE2_
  #endif
#else
// For everything else than MSVC
  #ifdef __SSE2__
    #define _YUVRGB_SSE2_
  #endif // __SSE2__
  #ifdef __AVX2__
	#define _YUVRGB_AVX2_
  #endif
#endif // _MSC_VER

uint8_t clamp(int16_t value)
{
	return value<0 ? 0 : (value>255 ? 255 : value);
}

// Definitions
//
// E'R, E'G, E'B, E'Y, E'Cb and E'Cr refer to the analog signals
// E'R, E'G, E'B and E'Y range is [0:1], while E'Cb and E'Cr range is [-0.5:0.5]
// R, G, B, Y, Cb and Cr refer to the digitalized values
// The digitalized values can use their full range ([0:255] for 8bit values),
// or a subrange (typically [16:235] for Y and [16:240] for CbCr).
// We assume here that RGB range is always [0:255], since it is the case for
// most digitalized images.
// For 8bit values :
// * Y = round((YMax-YMin)*E'Y + YMin)
// * Cb = round((CbRange)*E'Cb + 128)
// * Cr = round((CrRange)*E'Cr + 128)
// Where *Min and *Max are the range of each channel
//
// In the analog domain , the RGB to YCbCr transformation is defined as:
// * E'Y = Rf*E'R + Gf*E'G + Bf*E'B
// Where Rf, Gf and Bf are constants defined in each standard, with
// Rf + Gf + Bf = 1 (necessary to ensure that E'Y range is [0:1])
// * E'Cb = (E'B - E'Y) / CbNorm
// * E'Cr = (E'R - E'Y) / CrNorm
// Where CbNorm and CrNorm are constants, dependent of Rf, Gf, Bf, computed
// to normalize to a [-0.5:0.5] range : CbNorm=2*(1-Bf) and CrNorm=2*(1-Rf)
//
// Algorithms
//
// Most operations will be made in a fixed point format for speed, using
// N bits of precision. In next section the [x] convention is used for
// a fixed point rounded value, that is (int being the c type conversion)
// * [x] = int(x*(2^N)+0.5)
// N can be different for each factor, we simply use the highest value
// that will not overflow in 16 bits intermediate variables.
//.
// For RGB to YCbCr conversion, we start by generating a pseudo Y value
// (noted Y') in fixed point format, using the full range for now.
// * Y' = ([Rf]*R + [Gf]*G + [Bf]*B)>>N
// We can then compute Cb and Cr by
// * Cb = ((B - Y')*[CbRange/(255*CbNorm)])>>N + 128
// * Cr = ((R - Y')*[CrRange/(255*CrNorm)])>>N + 128
// And finally, we normalize Y to its digital range
// * Y = (Y'*[(YMax-YMin)/255])>>N + YMin
//
// For YCbCr to RGB conversion, we first compute the full range Y' value :
// * Y' = ((Y-YMin)*[255/(YMax-YMin)])>>N
// We can then compute B and R values by :
// * B = ((Cb-128)*[(255*CbNorm)/CbRange])>>N + Y'
// * R = ((Cr-128)*[(255*CrNorm)/CrRange])>>N + Y'
// And finally, for G we know that:
// * G = (Y' - (Rf*R + Bf*B)) / Gf
// From above:
// * G = (Y' - Rf * ((Cr-128)*(255*CrNorm)/CrRange + Y') - Bf * ((Cb-128)*(255*CbNorm)/CbRange + Y')) / Gf
// Since 1-Rf-Bf=Gf, we can take Y' out of the division by Gf, and we get:
// * G = Y' - (Cr-128)*Rf/Gf*(255*CrNorm)/CrRange - (Cb-128)*Bf/Gf*(255*CbNorm)/CbRange
// That we can compute, with fixed point arithmetic, by
// * G = Y' - ((Cr-128)*[Rf/Gf*(255*CrNorm)/CrRange] + (Cb-128)*[Bf/Gf*(255*CbNorm)/CbRange])>>N
//
// Note : in ITU-T T.871(JPEG), Y=Y', so that part could be optimized out


#define FIXED_POINT_VALUE(value, precision) ((int)(((value)*(1<<precision))+0.5))

// see above for description
typedef struct
{
	uint8_t r_factor;    // [Rf]
	uint8_t g_factor;    // [Rg]
	uint8_t b_factor;    // [Rb]
	uint8_t cb_factor;   // [CbRange/(255*CbNorm)]
	uint8_t cr_factor;   // [CrRange/(255*CrNorm)]
	uint8_t y_factor;    // [(YMax-YMin)/255]
	uint8_t y_offset;    // YMin
} RGB2YUVParam;

typedef struct
{
	uint8_t cb_factor;   // [(255*CbNorm)/CbRange]
	uint8_t cr_factor;   // [(255*CrNorm)/CrRange]
	uint8_t g_cb_factor; // [Bf/Gf*(255*CbNorm)/CbRange]
	uint8_t g_cr_factor; // [Rf/Gf*(255*CrNorm)/CrRange]
	uint8_t y_factor;    // [(YMax-YMin)/255]
	uint8_t y_offset;    // YMin
} YUV2RGBParam;

#define RGB2YUV_PARAM(Rf, Bf, YMin, YMax, CbCrRange) \
{.r_factor=FIXED_POINT_VALUE(Rf, 8), \
.g_factor=256-FIXED_POINT_VALUE(Rf, 8)-FIXED_POINT_VALUE(Bf, 8), \
.b_factor=FIXED_POINT_VALUE(Bf, 8), \
.cb_factor=FIXED_POINT_VALUE((CbCrRange/255.0)/(2.0*(1-Bf)), 8), \
.cr_factor=FIXED_POINT_VALUE((CbCrRange/255.0)/(2.0*(1-Rf)), 8), \
.y_factor=FIXED_POINT_VALUE((YMax-YMin)/255.0, 7), \
.y_offset=YMin}

#define YUV2RGB_PARAM(Rf, Bf, YMin, YMax, CbCrRange) \
{.cb_factor=FIXED_POINT_VALUE(255.0*(2.0*(1-Bf))/CbCrRange, 6), \
.cr_factor=FIXED_POINT_VALUE(255.0*(2.0*(1-Rf))/CbCrRange, 6), \
.g_cb_factor=FIXED_POINT_VALUE(Bf/(1.0-Bf-Rf)*255.0*(2.0*(1-Bf))/CbCrRange, 7), \
.g_cr_factor=FIXED_POINT_VALUE(Rf/(1.0-Bf-Rf)*255.0*(2.0*(1-Rf))/CbCrRange, 7), \
.y_factor=FIXED_POINT_VALUE(255.0/(YMax-YMin), 7), \
.y_offset=YMin}

static const RGB2YUVParam RGB2YUV[3] = {
	// ITU-T T.871 (JPEG)
	RGB2YUV_PARAM(0.299, 0.114, 0.0, 255.0, 255.0),
	// ITU-R BT.601-7
	RGB2YUV_PARAM(0.299, 0.114, 16.0, 235.0, 224.0),
	// ITU-R BT.709-6
	RGB2YUV_PARAM(0.2126, 0.0722, 16.0, 235.0, 224.0)
};

static const YUV2RGBParam YUV2RGB[3] = {
	// ITU-T T.871 (JPEG)
	YUV2RGB_PARAM(0.299, 0.114, 0.0, 255.0, 255.0),
	// ITU-R BT.601-7
	YUV2RGB_PARAM(0.299, 0.114, 16.0, 235.0, 224.0),
	// ITU-R BT.709-6
	YUV2RGB_PARAM(0.2126, 0.0722, 16.0, 235.0, 224.0)
};


#define RGB24_YUV420_CONVERT_TWO_LINE \
{ \
	uint8_t y_tmp; \
	int16_t u_tmp, v_tmp; \
	\
	y_tmp = (param->r_factor*rgb_ptr1[0] + param->g_factor*rgb_ptr1[1] + param->b_factor*rgb_ptr1[2])>>8; \
	u_tmp = rgb_ptr1[2]-y_tmp; \
	v_tmp = rgb_ptr1[0]-y_tmp; \
	y_ptr1[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr1[3] + param->g_factor*rgb_ptr1[4] + param->b_factor*rgb_ptr1[5])>>8; \
	u_tmp += rgb_ptr1[5]-y_tmp; \
	v_tmp += rgb_ptr1[3]-y_tmp; \
	y_ptr1[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr2[0] + param->g_factor*rgb_ptr2[1] + param->b_factor*rgb_ptr2[2])>>8; \
	u_tmp += rgb_ptr2[2]-y_tmp; \
	v_tmp += rgb_ptr2[0]-y_tmp; \
	y_ptr2[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr2[3] + param->g_factor*rgb_ptr2[4] + param->b_factor*rgb_ptr2[5])>>8; \
	u_tmp += rgb_ptr2[5]-y_tmp; \
	v_tmp += rgb_ptr2[3]-y_tmp; \
	y_ptr2[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	u_ptr[0] = (((u_tmp>>2)*param->cb_factor)>>8) + 128; \
	v_ptr[0] = (((v_tmp>>2)*param->cr_factor)>>8) + 128; \
	\
	rgb_ptr1 += 6; \
	rgb_ptr2 += 6; \
	y_ptr1 += 2; \
	y_ptr2 += 2; \
	u_ptr += 1; \
	v_ptr += 1; \
}

void rgb24_yuv420_std(
	uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB24_YUV420_CONVERT_TWO_LINE;
		}
	}
}

static inline uint8_t av_clip_uint8(int a)
{
	if (a&(~0xFF)) return (-a)>>31;
	else           return a;
}

void rgb24_yuv420_std_ffmpeg(
	uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{	
	yuv_type = yuv_type;

#define RGB2YUV_SHIFT 15
#define BY ( (int) (0.114 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define BV (-(int) (0.081 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define BU ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GY ( (int) (0.587 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GV (-(int) (0.419 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define GU (-(int) (0.331 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RY ( (int) (0.299 * 219 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RV ( (int) (0.500 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))
#define RU (-(int) (0.169 * 224 / 255 * (1 << RGB2YUV_SHIFT) + 0.5))

	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			uint8_t y_tmp;
			int16_t u_tmp, v_tmp;

			y_tmp = av_clip_uint8((RY * rgb_ptr1[0] + GY * rgb_ptr1[1] + BY * rgb_ptr1[2] + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			u_tmp = av_clip_uint8((RU * rgb_ptr1[0] + GU * rgb_ptr1[1] + BU * rgb_ptr1[2] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			v_tmp = av_clip_uint8((RV * rgb_ptr1[0] + GV * rgb_ptr1[1] + BV * rgb_ptr1[2] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			y_ptr1[0]= (y_tmp);

			y_tmp = av_clip_uint8((RY * rgb_ptr1[3] + GY * rgb_ptr1[4] + BY * rgb_ptr1[5] + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			u_tmp += av_clip_uint8((RU * rgb_ptr1[3] + GU * rgb_ptr1[4] + BU * rgb_ptr1[5] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			v_tmp += av_clip_uint8((RV * rgb_ptr1[3] + GV * rgb_ptr1[4] + BV * rgb_ptr1[5] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			y_ptr1[1]= (y_tmp);

			y_tmp = av_clip_uint8((RY * rgb_ptr2[0] + GY * rgb_ptr2[1] + BY * rgb_ptr2[2] + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			u_tmp += av_clip_uint8((RU * rgb_ptr2[0] + GU * rgb_ptr2[1] + BU * rgb_ptr2[2] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			v_tmp += av_clip_uint8((RV * rgb_ptr2[0] + GV * rgb_ptr2[1] + BV * rgb_ptr2[2] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			y_ptr2[0]= (y_tmp);

			y_tmp = av_clip_uint8((RY * rgb_ptr2[3] + GY * rgb_ptr2[4] + BY * rgb_ptr2[5] + ( 33 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			u_tmp += av_clip_uint8((RU * rgb_ptr2[3] + GU * rgb_ptr2[4] + BU * rgb_ptr2[5] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			v_tmp += av_clip_uint8((RV * rgb_ptr2[3] + GV * rgb_ptr2[4] + BV * rgb_ptr2[5] + ( 257 << (RGB2YUV_SHIFT - 1))) >> RGB2YUV_SHIFT);
			y_ptr2[1]= (y_tmp);

			u_ptr[0] = u_tmp / 4;
			v_ptr[0] = v_tmp / 4;

			rgb_ptr1 += 6;
			rgb_ptr2 += 6;
			y_ptr1 += 2;
			y_ptr2 += 2;
			u_ptr += 1;
			v_ptr += 1;
		}
	}
}

#define RGB32_YUV420_CONVERT_TWO_LINE \
{ \
	uint8_t y_tmp; \
	int16_t u_tmp, v_tmp; \
	\
	y_tmp = (param->r_factor*rgb_ptr1[0] + param->g_factor*rgb_ptr1[1] + param->b_factor*rgb_ptr1[2])>>8; \
	u_tmp = rgb_ptr1[2]-y_tmp; \
	v_tmp = rgb_ptr1[0]-y_tmp; \
	y_ptr1[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr1[4] + param->g_factor*rgb_ptr1[5] + param->b_factor*rgb_ptr1[6])>>8; \
	u_tmp += rgb_ptr1[6]-y_tmp; \
	v_tmp += rgb_ptr1[4]-y_tmp; \
	y_ptr1[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr2[0] + param->g_factor*rgb_ptr2[1] + param->b_factor*rgb_ptr2[2])>>8; \
	u_tmp += rgb_ptr2[2]-y_tmp; \
	v_tmp += rgb_ptr2[0]-y_tmp; \
	y_ptr2[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	y_tmp = (param->r_factor*rgb_ptr2[4] + param->g_factor*rgb_ptr2[5] + param->b_factor*rgb_ptr2[6])>>8; \
	u_tmp += rgb_ptr2[6]-y_tmp; \
	v_tmp += rgb_ptr2[4]-y_tmp; \
	y_ptr2[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	\
	u_ptr[0] = (((u_tmp>>2)*param->cb_factor)>>8) + 128; \
	v_ptr[0] = (((v_tmp>>2)*param->cr_factor)>>8) + 128; \
	\
	rgb_ptr1 += 8; \
	rgb_ptr2 += 8; \
	y_ptr1 += 2; \
	y_ptr2 += 2; \
	u_ptr += 1; \
	v_ptr += 1; \
}

void rgb32_yuv420_std(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGBA+y*RGBA_stride,
			*rgb_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUV420_CONVERT_TWO_LINE;
		}
	}
}


void yuv420_rgb24_std(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *U, const uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			int8_t u_tmp, v_tmp;
			u_tmp = u_ptr[0]-128;
			v_tmp = v_ptr[0]-128;
			
			//compute Cb Cr color offsets, common to four pixels
			int16_t b_cb_offset, r_cr_offset, g_cbcr_offset;
			b_cb_offset = (param->cb_factor*u_tmp)>>6;
			r_cr_offset = (param->cr_factor*v_tmp)>>6;
			g_cbcr_offset = (param->g_cb_factor*u_tmp + param->g_cr_factor*v_tmp)>>7;
			
			int16_t y_tmp;
			y_tmp = (param->y_factor*(y_ptr1[0]-param->y_offset))>>7;
			rgb_ptr1[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr1[1]-param->y_offset))>>7;
			rgb_ptr1[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[5] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[0]-param->y_offset))>>7;
			rgb_ptr2[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[1]-param->y_offset))>>7;
			rgb_ptr2[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[5] = clamp(y_tmp + b_cb_offset);
			
			rgb_ptr1 += 6;
			rgb_ptr2 += 6;
			y_ptr1 += 2;
			y_ptr2 += 2;
			u_ptr += 1;
			v_ptr += 1;
		}
	}
}

void nv12_rgb24_std(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			int8_t u_tmp, v_tmp;
			u_tmp = uv_ptr[0]-128;
			v_tmp = uv_ptr[1]-128;
			
			//compute Cb Cr color offsets, common to four pixels
			int16_t b_cb_offset, r_cr_offset, g_cbcr_offset;
			b_cb_offset = (param->cb_factor*u_tmp)>>6;
			r_cr_offset = (param->cr_factor*v_tmp)>>6;
			g_cbcr_offset = (param->g_cb_factor*u_tmp + param->g_cr_factor*v_tmp)>>7;
			
			int16_t y_tmp;
			y_tmp = (param->y_factor*(y_ptr1[0]-param->y_offset))>>7;
			rgb_ptr1[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr1[1]-param->y_offset))>>7;
			rgb_ptr1[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[5] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[0]-param->y_offset))>>7;
			rgb_ptr2[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[1]-param->y_offset))>>7;
			rgb_ptr2[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[5] = clamp(y_tmp + b_cb_offset);
			
			rgb_ptr1 += 6;
			rgb_ptr2 += 6;
			y_ptr1 += 2;
			y_ptr2 += 2;
			uv_ptr += 2;
		}
	}
}

void nv21_rgb24_std(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			int8_t u_tmp, v_tmp;
			u_tmp = uv_ptr[1]-128;
			v_tmp = uv_ptr[0]-128;
			
			//compute Cb Cr color offsets, common to four pixels
			int16_t b_cb_offset, r_cr_offset, g_cbcr_offset;
			b_cb_offset = (param->cb_factor*u_tmp)>>6;
			r_cr_offset = (param->cr_factor*v_tmp)>>6;
			g_cbcr_offset = (param->g_cb_factor*u_tmp + param->g_cr_factor*v_tmp)>>7;
			
			int16_t y_tmp;
			y_tmp = (param->y_factor*(y_ptr1[0]-param->y_offset))>>7;
			rgb_ptr1[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr1[1]-param->y_offset))>>7;
			rgb_ptr1[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr1[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr1[5] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[0]-param->y_offset))>>7;
			rgb_ptr2[0] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[1] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[2] = clamp(y_tmp + b_cb_offset);
			
			y_tmp = (param->y_factor*(y_ptr2[1]-param->y_offset))>>7;
			rgb_ptr2[3] = clamp(y_tmp + r_cr_offset);
			rgb_ptr2[4] = clamp(y_tmp - g_cbcr_offset);
			rgb_ptr2[5] = clamp(y_tmp + b_cb_offset);
			
			rgb_ptr1 += 6;
			rgb_ptr2 += 6;
			y_ptr1 += 2;
			y_ptr2 += 2;
			uv_ptr += 2;
		}
	}
}


#ifdef _YUVRGB_SSE2_

//see rgb.txt
#define UNPACK_RGB24_32_STEP(RS1, RS2, RS3, RS4, RS5, RS6, RD1, RD2, RD3, RD4, RD5, RD6) \
RD1 = _mm_unpacklo_epi8(RS1, RS4); \
RD2 = _mm_unpackhi_epi8(RS1, RS4); \
RD3 = _mm_unpacklo_epi8(RS2, RS5); \
RD4 = _mm_unpackhi_epi8(RS2, RS5); \
RD5 = _mm_unpacklo_epi8(RS3, RS6); \
RD6 = _mm_unpackhi_epi8(RS3, RS6);

#define RGB2YUV_16(R, G, B, Y, U, V) \
Y = _mm_add_epi16(_mm_mullo_epi16(R, _mm_set1_epi16(param->r_factor)), \
                  _mm_mullo_epi16(G, _mm_set1_epi16(param->g_factor))); \
Y = _mm_add_epi16(Y, _mm_mullo_epi16(B, _mm_set1_epi16(param->b_factor))); \
Y = _mm_srli_epi16(Y, 8); \
U = _mm_mullo_epi16(_mm_sub_epi16(B, Y), _mm_set1_epi16(param->cb_factor)); \
U = _mm_add_epi16(_mm_srai_epi16(U, 8), _mm_set1_epi16(128)); \
V = _mm_mullo_epi16(_mm_sub_epi16(R, Y), _mm_set1_epi16(param->cr_factor)); \
V = _mm_add_epi16(_mm_srai_epi16(V, 8), _mm_set1_epi16(128)); \
Y = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(Y, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset));

#define RGB2YUV_32 \
	__m128i r_16, g_16, b_16; \
	__m128i y1_16, y2_16, cb1_16, cb2_16, cr1_16, cr2_16, Y, cb, cr; \
	__m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6; \
	__m128i rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+16)), \
		rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+32)), \
		rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr2)), \
		rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2+16)), \
		rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+32)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_sub_epi16(b_16, y1_16); \
	cr1_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb4, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y1_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb4, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb1_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr1_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1+48)), \
	rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+64)), \
	rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+80)), \
	rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr2+48)), \
	rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2+64)), \
	rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+80)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_sub_epi16(b_16, y1_16); \
	cr2_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb4, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1+16), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y1_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb4, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2+16), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb2_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr2_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm_packus_epi16(cb1_16, cb2_16); \
	cr = _mm_packus_epi16(cr1_16, cr2_16); \
	SAVE_SI128((__m128i*)(u_ptr), cb); \
	SAVE_SI128((__m128i*)(v_ptr), cr);


void rgb24_yuv420_sse(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	width = width;

	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(Y_stride-31); x+=32)
		{
			RGB2YUV_32
			
			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB24_YUV420_CONVERT_TWO_LINE;
		}		
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void rgb24_yuv420_sseu(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			RGB2YUV_32
			
			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB24_YUV420_CONVERT_TWO_LINE;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}


// see rgba.txt
#define UNPACK_RGB32_32_STEP(RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8, RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8) \
RD1 = _mm_unpacklo_epi8(RS1, RS5); \
RD2 = _mm_unpackhi_epi8(RS1, RS5); \
RD3 = _mm_unpacklo_epi8(RS2, RS6); \
RD4 = _mm_unpackhi_epi8(RS2, RS6); \
RD5 = _mm_unpacklo_epi8(RS3, RS7); \
RD6 = _mm_unpackhi_epi8(RS3, RS7); \
RD7 = _mm_unpacklo_epi8(RS4, RS8); \
RD8 = _mm_unpackhi_epi8(RS4, RS8);


#define RGBA2YUV_32 \
	__m128i r_16, g_16, b_16; \
	__m128i y1_16, y2_16, cb1_16, cb2_16, cr1_16, cr2_16, Y, cb, cr; \
	__m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8; \
	__m128i rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+16)), \
		rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+32)), \
		rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr1+48)), \
		rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2)), \
		rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+16)), \
		rgb7 = LOAD_SI128((const __m128i*)(rgb_ptr2+32)), \
		rgb8 = LOAD_SI128((const __m128i*)(rgb_ptr2+48)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_sub_epi16(b_16, y1_16); \
	cr1_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y1_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb1_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr1_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI128((const __m128i*)(rgb_ptr1+64)), \
	rgb2 = LOAD_SI128((const __m128i*)(rgb_ptr1+80)), \
	rgb3 = LOAD_SI128((const __m128i*)(rgb_ptr1+96)), \
	rgb4 = LOAD_SI128((const __m128i*)(rgb_ptr1+112)), \
	rgb5 = LOAD_SI128((const __m128i*)(rgb_ptr2+64)), \
	rgb6 = LOAD_SI128((const __m128i*)(rgb_ptr2+80)), \
	rgb7 = LOAD_SI128((const __m128i*)(rgb_ptr2+96)), \
	rgb8 = LOAD_SI128((const __m128i*)(rgb_ptr2+112)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_sub_epi16(b_16, y1_16); \
	cr2_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1+16), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y1_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2+16), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb2_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr2_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm_packus_epi16(cb1_16, cb2_16); \
	cr = _mm_packus_epi16(cr1_16, cr2_16); \
	SAVE_SI128((__m128i*)(u_ptr), cb); \
	SAVE_SI128((__m128i*)(v_ptr), cr);

void rgb32_yuv420_sse(uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGBA+y*RGBA_stride,
			*rgb_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			RGBA2YUV_32
			
			rgb_ptr1+=128;
			rgb_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUV420_CONVERT_TWO_LINE;
		}		
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void rgb32_yuv420_sseu(uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGBA+y*RGBA_stride,
			*rgb_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			RGBA2YUV_32
			
			rgb_ptr1+=128;
			rgb_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUV420_CONVERT_TWO_LINE;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}


#define RGB32_YUVA420_CONVERT_TWO_LINE \
{ \
	uint8_t y_tmp; \
	int16_t u_tmp, v_tmp; \
	\
	y_tmp = (param->r_factor*rgba_ptr1[0] + param->g_factor*rgba_ptr1[1] + param->b_factor*rgba_ptr1[2])>>8; \
	u_tmp = rgba_ptr1[2]-y_tmp; \
	v_tmp = rgba_ptr1[0]-y_tmp; \
	y_ptr1[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	a_ptr1[0]=rgba_ptr1[3]; \
	\
	y_tmp = (param->r_factor*rgba_ptr1[4] + param->g_factor*rgba_ptr1[5] + param->b_factor*rgba_ptr1[6])>>8; \
	u_tmp += rgba_ptr1[6]-y_tmp; \
	v_tmp += rgba_ptr1[4]-y_tmp; \
	y_ptr1[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	a_ptr1[1]=rgba_ptr1[7]; \
	\
	y_tmp = (param->r_factor*rgba_ptr2[0] + param->g_factor*rgba_ptr2[1] + param->b_factor*rgba_ptr2[2])>>8; \
	u_tmp += rgba_ptr2[2]-y_tmp; \
	v_tmp += rgba_ptr2[0]-y_tmp; \
	y_ptr2[0]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	a_ptr2[0]=rgba_ptr2[3]; \
	\
	y_tmp = (param->r_factor*rgba_ptr2[4] + param->g_factor*rgba_ptr2[5] + param->b_factor*rgba_ptr2[6])>>8; \
	u_tmp += rgba_ptr2[6]-y_tmp; \
	v_tmp += rgba_ptr2[4]-y_tmp; \
	y_ptr2[1]=((y_tmp*param->y_factor)>>7) + param->y_offset; \
	a_ptr2[1]=rgba_ptr2[7]; \
	\
	u_ptr[0] = (((u_tmp>>2)*param->cb_factor)>>8) + 128; \
	v_ptr[0] = (((v_tmp>>2)*param->cr_factor)>>8) + 128; \
	\
	rgba_ptr1 += 8; \
	rgba_ptr2 += 8; \
	y_ptr1 += 2; \
	y_ptr2 += 2; \
	u_ptr += 1; \
	v_ptr += 1; \
	a_ptr1 += 2; \
	a_ptr2 += 2; \
}


void rgb32_yuva420_std(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint8_t *A, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgba_ptr1=RGBA+y*RGBA_stride,
			*rgba_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride,
			*a_ptr1=A+y*Y_stride,
			*a_ptr2=A+(y+1)*Y_stride;
		
		for(x=0; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUVA420_CONVERT_TWO_LINE;
		}
	}
}


#define RGBA2YUVA_32 \
	__m128i r_16, g_16, b_16; \
	__m128i y1_16, y2_16, cb1_16, cb2_16, cr1_16, cr2_16, Y, cb, cr, A; \
	__m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8; \
	__m128i rgb1 = LOAD_SI128((const __m128i*)(rgba_ptr1)), \
		rgb2 = LOAD_SI128((const __m128i*)(rgba_ptr1+16)), \
		rgb3 = LOAD_SI128((const __m128i*)(rgba_ptr1+32)), \
		rgb4 = LOAD_SI128((const __m128i*)(rgba_ptr1+48)), \
		rgb5 = LOAD_SI128((const __m128i*)(rgba_ptr2)), \
		rgb6 = LOAD_SI128((const __m128i*)(rgba_ptr2+16)), \
		rgb7 = LOAD_SI128((const __m128i*)(rgba_ptr2+32)), \
		rgb8 = LOAD_SI128((const __m128i*)(rgba_ptr2+48)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_sub_epi16(b_16, y1_16); \
	cr1_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y1_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb1_16 = _mm_add_epi16(cb1_16, _mm_sub_epi16(b_16, y2_16)); \
	cr1_16 = _mm_add_epi16(cr1_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2), Y); \
	A = _mm_unpacklo_epi8(rgb4, rgb8); \
	SAVE_SI128((__m128i*)(a_ptr1), A); \
	A = _mm_unpackhi_epi8(rgb4, rgb8); \
	SAVE_SI128((__m128i*)(a_ptr2), A); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb1_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr1_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr1_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI128((const __m128i*)(rgba_ptr1+64)), \
	rgb2 = LOAD_SI128((const __m128i*)(rgba_ptr1+80)), \
	rgb3 = LOAD_SI128((const __m128i*)(rgba_ptr1+96)), \
	rgb4 = LOAD_SI128((const __m128i*)(rgba_ptr1+112)), \
	rgb5 = LOAD_SI128((const __m128i*)(rgba_ptr2+64)), \
	rgb6 = LOAD_SI128((const __m128i*)(rgba_ptr2+80)), \
	rgb7 = LOAD_SI128((const __m128i*)(rgba_ptr2+96)), \
	rgb8 = LOAD_SI128((const __m128i*)(rgba_ptr2+112)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_32_STEP(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_32_STEP(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_16 = _mm_unpacklo_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_sub_epi16(b_16, y1_16); \
	cr2_16 = _mm_sub_epi16(r_16, y1_16); \
	r_16 = _mm_unpacklo_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpacklo_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpacklo_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr1+16), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_16 = _mm_unpackhi_epi8(rgb1, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb2, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb3, _mm_setzero_si128()); \
	y1_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y1_16 = _mm_add_epi16(y1_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y1_16 = _mm_srli_epi16(y1_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y1_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y1_16)); \
	r_16 = _mm_unpackhi_epi8(rgb5, _mm_setzero_si128()); \
	g_16 = _mm_unpackhi_epi8(rgb6, _mm_setzero_si128()); \
	b_16 = _mm_unpackhi_epi8(rgb7, _mm_setzero_si128()); \
	y2_16 = _mm_add_epi16(_mm_mullo_epi16(r_16, _mm_set1_epi16(param->r_factor)), \
		_mm_mullo_epi16(g_16, _mm_set1_epi16(param->g_factor))); \
	y2_16 = _mm_add_epi16(y2_16, _mm_mullo_epi16(b_16, _mm_set1_epi16(param->b_factor))); \
	y2_16 = _mm_srli_epi16(y2_16, 8); \
	cb2_16 = _mm_add_epi16(cb2_16, _mm_sub_epi16(b_16, y2_16)); \
	cr2_16 = _mm_add_epi16(cr2_16, _mm_sub_epi16(r_16, y2_16)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y1_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	y2_16 = _mm_add_epi16(_mm_srli_epi16(_mm_mullo_epi16(y2_16, _mm_set1_epi16(param->y_factor)), 7), _mm_set1_epi16(param->y_offset)); \
	Y = _mm_packus_epi16(y1_16, y2_16); \
	Y = _mm_unpackhi_epi8(_mm_slli_si128(Y, 8), Y); \
	SAVE_SI128((__m128i*)(y_ptr2+16), Y); \
	A = _mm_unpacklo_epi8(rgb4, rgb8); \
	SAVE_SI128((__m128i*)(a_ptr1+16), A); \
	A = _mm_unpackhi_epi8(rgb4, rgb8); \
	SAVE_SI128((__m128i*)(a_ptr2+16), A); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cb2_16, 2), _mm_set1_epi16(param->cb_factor)), 8), _mm_set1_epi16(128)); \
	cr2_16 = _mm_add_epi16(_mm_srai_epi16(_mm_mullo_epi16(_mm_srai_epi16(cr2_16, 2), _mm_set1_epi16(param->cr_factor)), 8), _mm_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm_packus_epi16(cb1_16, cb2_16); \
	cr = _mm_packus_epi16(cr1_16, cr2_16); \
	SAVE_SI128((__m128i*)(u_ptr), cb); \
	SAVE_SI128((__m128i*)(v_ptr), cr);

void rgba32_yuva420_sse(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint8_t *A, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_store_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgba_ptr1=RGBA+y*RGBA_stride,
			*rgba_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride,
			*a_ptr1=A+y*Y_stride,
			*a_ptr2=A+(y+1)*Y_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			RGBA2YUVA_32
			
			rgba_ptr1+=128;
			rgba_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			a_ptr1+=32;
			a_ptr2+=32;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUVA420_CONVERT_TWO_LINE;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void rgba32_yuva420_sseu(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint8_t *A, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert((width % 2 + height % 2) == 0);

	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgba_ptr1=RGBA+y*RGBA_stride,
			*rgba_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride,
			*a_ptr1=A+y*Y_stride,
			*a_ptr2=A+(y+1)*Y_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			RGBA2YUVA_32
			
			rgba_ptr1+=128;
			rgba_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			a_ptr1+=32;
			a_ptr2+=32;
		}

		for(; x<(width-1); x+=2)
		{
			// compute yuv for the four pixels, u and v values are summed
			RGB32_YUVA420_CONVERT_TWO_LINE;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

#endif  // _YUVRGB_SSE2_

#ifdef _YUVRGB_SSE2_

#define UV2RGB_16(U,V,R1,G1,B1,R2,G2,B2) \
	r_tmp = _mm_srai_epi16(_mm_mullo_epi16(V, _mm_set1_epi16(param->cr_factor)), 6); \
	g_tmp = _mm_srai_epi16(_mm_add_epi16( \
		_mm_mullo_epi16(U, _mm_set1_epi16(param->g_cb_factor)), \
		_mm_mullo_epi16(V, _mm_set1_epi16(param->g_cr_factor))), 7); \
	b_tmp = _mm_srai_epi16(_mm_mullo_epi16(U, _mm_set1_epi16(param->cb_factor)), 6); \
	R1 = _mm_unpacklo_epi16(r_tmp, r_tmp); \
	G1 = _mm_unpacklo_epi16(g_tmp, g_tmp); \
	B1 = _mm_unpacklo_epi16(b_tmp, b_tmp); \
	R2 = _mm_unpackhi_epi16(r_tmp, r_tmp); \
	G2 = _mm_unpackhi_epi16(g_tmp, g_tmp); \
	B2 = _mm_unpackhi_epi16(b_tmp, b_tmp); \

#define ADD_Y2RGB_16(Y1,Y2,R1,G1,B1,R2,G2,B2) \
	Y1 = _mm_srli_epi16(_mm_mullo_epi16(Y1, _mm_set1_epi16(param->y_factor)), 7); \
	Y2 = _mm_srli_epi16(_mm_mullo_epi16(Y2, _mm_set1_epi16(param->y_factor)), 7); \
	\
	R1 = _mm_add_epi16(Y1, R1); \
	G1 = _mm_sub_epi16(Y1, G1); \
	B1 = _mm_add_epi16(Y1, B1); \
	R2 = _mm_add_epi16(Y2, R2); \
	G2 = _mm_sub_epi16(Y2, G2); \
	B2 = _mm_add_epi16(Y2, B2); \

#define PACK_RGB24_32_STEP(RS1, RS2, RS3, RS4, RS5, RS6, RD1, RD2, RD3, RD4, RD5, RD6) \
RD1 = _mm_packus_epi16(_mm_and_si128(RS1,_mm_set1_epi16(0xFF)), _mm_and_si128(RS2,_mm_set1_epi16(0xFF))); \
RD2 = _mm_packus_epi16(_mm_and_si128(RS3,_mm_set1_epi16(0xFF)), _mm_and_si128(RS4,_mm_set1_epi16(0xFF))); \
RD3 = _mm_packus_epi16(_mm_and_si128(RS5,_mm_set1_epi16(0xFF)), _mm_and_si128(RS6,_mm_set1_epi16(0xFF))); \
RD4 = _mm_packus_epi16(_mm_srli_epi16(RS1,8), _mm_srli_epi16(RS2,8)); \
RD5 = _mm_packus_epi16(_mm_srli_epi16(RS3,8), _mm_srli_epi16(RS4,8)); \
RD6 = _mm_packus_epi16(_mm_srli_epi16(RS5,8), _mm_srli_epi16(RS6,8)); \

#define PACK_RGB24_32(R1, R2, G1, G2, B1, B2, RGB1, RGB2, RGB3, RGB4, RGB5, RGB6) \
PACK_RGB24_32_STEP(R1, R2, G1, G2, B1, B2, RGB1, RGB2, RGB3, RGB4, RGB5, RGB6) \
PACK_RGB24_32_STEP(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
PACK_RGB24_32_STEP(R1, R2, G1, G2, B1, B2, RGB1, RGB2, RGB3, RGB4, RGB5, RGB6) \
PACK_RGB24_32_STEP(RGB1, RGB2, RGB3, RGB4, RGB5, RGB6, R1, R2, G1, G2, B1, B2) \
PACK_RGB24_32_STEP(R1, R2, G1, G2, B1, B2, RGB1, RGB2, RGB3, RGB4, RGB5, RGB6) \

#define LOAD_UV_PLANAR \
	__m128i u = LOAD_SI128((const __m128i*)(u_ptr)); \
	__m128i v = LOAD_SI128((const __m128i*)(v_ptr)); \

#define LOAD_UV_NV12 \
	__m128i uv1 = LOAD_SI128((const __m128i*)(uv_ptr)); \
	__m128i uv2 = LOAD_SI128((const __m128i*)(uv_ptr+16)); \
	__m128i u = _mm_packus_epi16(_mm_and_si128(uv1, _mm_set1_epi16(255)), _mm_and_si128(uv2, _mm_set1_epi16(255))); \
	uv1 = _mm_srli_epi16(uv1, 8); \
	uv2 = _mm_srli_epi16(uv2, 8); \
	__m128i v = _mm_packus_epi16(_mm_and_si128(uv1, _mm_set1_epi16(255)), _mm_and_si128(uv2, _mm_set1_epi16(255))); \

#define LOAD_UV_NV21 \
	__m128i uv1 = LOAD_SI128((const __m128i*)(uv_ptr)); \
	__m128i uv2 = LOAD_SI128((const __m128i*)(uv_ptr+16)); \
	__m128i v = _mm_packus_epi16(_mm_and_si128(uv1, _mm_set1_epi16(255)), _mm_and_si128(uv2, _mm_set1_epi16(255))); \
	uv1 = _mm_srli_epi16(uv1, 8); \
	uv2 = _mm_srli_epi16(uv2, 8); \
	__m128i u = _mm_packus_epi16(_mm_and_si128(uv1, _mm_set1_epi16(255)), _mm_and_si128(uv2, _mm_set1_epi16(255))); \

#define YUV2RGB_32 \
	__m128i r_tmp, g_tmp, b_tmp; \
	__m128i r_16_1, g_16_1, b_16_1, r_16_2, g_16_2, b_16_2; \
	__m128i r_uv_16_1, g_uv_16_1, b_uv_16_1, r_uv_16_2, g_uv_16_2, b_uv_16_2; \
	__m128i y_16_1, y_16_2; \
	\
	u = _mm_add_epi8(u, _mm_set1_epi8(-128)); \
	v = _mm_add_epi8(v, _mm_set1_epi8(-128)); \
	\
	/* process first 16 pixels of first line */\
	__m128i u_16 = _mm_srai_epi16(_mm_unpacklo_epi8(u, u), 8); \
	__m128i v_16 = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8); \
	\
	UV2RGB_16(u_16, v_16, r_uv_16_1, g_uv_16_1, b_uv_16_1, r_uv_16_2, g_uv_16_2, b_uv_16_2) \
	r_16_1=r_uv_16_1; g_16_1=g_uv_16_1; b_16_1=b_uv_16_1; \
	r_16_2=r_uv_16_2; g_16_2=g_uv_16_2; b_16_2=b_uv_16_2; \
	\
	__m128i y = LOAD_SI128((const __m128i*)(y_ptr1)); \
	y = _mm_subs_epu8(y, _mm_set1_epi8(param->y_offset)); \
	y_16_1 = _mm_unpacklo_epi8(y, _mm_setzero_si128()); \
	y_16_2 = _mm_unpackhi_epi8(y, _mm_setzero_si128()); \
	\
	ADD_Y2RGB_16(y_16_1, y_16_2, r_16_1, g_16_1, b_16_1, r_16_2, g_16_2, b_16_2) \
	\
	__m128i r_8_11 = _mm_packus_epi16(r_16_1, r_16_2); \
	__m128i g_8_11 = _mm_packus_epi16(g_16_1, g_16_2); \
	__m128i b_8_11 = _mm_packus_epi16(b_16_1, b_16_2); \
	\
	/* process first 16 pixels of second line */\
	r_16_1=r_uv_16_1; g_16_1=g_uv_16_1; b_16_1=b_uv_16_1; \
	r_16_2=r_uv_16_2; g_16_2=g_uv_16_2; b_16_2=b_uv_16_2; \
	\
	y = LOAD_SI128((const __m128i*)(y_ptr2)); \
	y = _mm_subs_epu8(y, _mm_set1_epi8(param->y_offset)); \
	y_16_1 = _mm_unpacklo_epi8(y, _mm_setzero_si128()); \
	y_16_2 = _mm_unpackhi_epi8(y, _mm_setzero_si128()); \
	\
	ADD_Y2RGB_16(y_16_1, y_16_2, r_16_1, g_16_1, b_16_1, r_16_2, g_16_2, b_16_2) \
	\
	__m128i r_8_21 = _mm_packus_epi16(r_16_1, r_16_2); \
	__m128i g_8_21 = _mm_packus_epi16(g_16_1, g_16_2); \
	__m128i b_8_21 = _mm_packus_epi16(b_16_1, b_16_2); \
	\
	/* process last 16 pixels of first line */\
	u_16 = _mm_srai_epi16(_mm_unpackhi_epi8(u, u), 8); \
	v_16 = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 8); \
	\
	UV2RGB_16(u_16, v_16, r_uv_16_1, g_uv_16_1, b_uv_16_1, r_uv_16_2, g_uv_16_2, b_uv_16_2) \
	r_16_1=r_uv_16_1; g_16_1=g_uv_16_1; b_16_1=b_uv_16_1; \
	r_16_2=r_uv_16_2; g_16_2=g_uv_16_2; b_16_2=b_uv_16_2; \
	\
	y = LOAD_SI128((const __m128i*)(y_ptr1+16)); \
	y = _mm_subs_epu8(y, _mm_set1_epi8(param->y_offset)); \
	y_16_1 = _mm_unpacklo_epi8(y, _mm_setzero_si128()); \
	y_16_2 = _mm_unpackhi_epi8(y, _mm_setzero_si128()); \
	\
	ADD_Y2RGB_16(y_16_1, y_16_2, r_16_1, g_16_1, b_16_1, r_16_2, g_16_2, b_16_2) \
	\
	__m128i r_8_12 = _mm_packus_epi16(r_16_1, r_16_2); \
	__m128i g_8_12 = _mm_packus_epi16(g_16_1, g_16_2); \
	__m128i b_8_12 = _mm_packus_epi16(b_16_1, b_16_2); \
	\
	/* process last 16 pixels of second line */\
	r_16_1=r_uv_16_1; g_16_1=g_uv_16_1; b_16_1=b_uv_16_1; \
	r_16_2=r_uv_16_2; g_16_2=g_uv_16_2; b_16_2=b_uv_16_2; \
	\
	y = LOAD_SI128((const __m128i*)(y_ptr2+16)); \
	y = _mm_subs_epu8(y, _mm_set1_epi8(param->y_offset)); \
	y_16_1 = _mm_unpacklo_epi8(y, _mm_setzero_si128()); \
	y_16_2 = _mm_unpackhi_epi8(y, _mm_setzero_si128()); \
	\
	ADD_Y2RGB_16(y_16_1, y_16_2, r_16_1, g_16_1, b_16_1, r_16_2, g_16_2, b_16_2) \
	\
	__m128i r_8_22 = _mm_packus_epi16(r_16_1, r_16_2); \
	__m128i g_8_22 = _mm_packus_epi16(g_16_1, g_16_2); \
	__m128i b_8_22 = _mm_packus_epi16(b_16_1, b_16_2); \
	\
	__m128i rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6; \
	\
	PACK_RGB24_32(r_8_11, r_8_12, g_8_11, g_8_12, b_8_11, b_8_12, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6) \
	SAVE_SI128((__m128i*)(rgb_ptr1), rgb_1); \
	SAVE_SI128((__m128i*)(rgb_ptr1+16), rgb_2); \
	SAVE_SI128((__m128i*)(rgb_ptr1+32), rgb_3); \
	SAVE_SI128((__m128i*)(rgb_ptr1+48), rgb_4); \
	SAVE_SI128((__m128i*)(rgb_ptr1+64), rgb_5); \
	SAVE_SI128((__m128i*)(rgb_ptr1+80), rgb_6); \
	\
	PACK_RGB24_32(r_8_21, r_8_22, g_8_21, g_8_22, b_8_21, b_8_22, rgb_1, rgb_2, rgb_3, rgb_4, rgb_5, rgb_6) \
	SAVE_SI128((__m128i*)(rgb_ptr2), rgb_1); \
	SAVE_SI128((__m128i*)(rgb_ptr2+16), rgb_2); \
	SAVE_SI128((__m128i*)(rgb_ptr2+32), rgb_3); \
	SAVE_SI128((__m128i*)(rgb_ptr2+48), rgb_4); \
	SAVE_SI128((__m128i*)(rgb_ptr2+64), rgb_5); \
	SAVE_SI128((__m128i*)(rgb_ptr2+80), rgb_6); \

#define YUV2RGB_32_PLANAR \
	LOAD_UV_PLANAR \
	YUV2RGB_32

#define YUV2RGB_32_NV12 \
	LOAD_UV_NV12 \
	YUV2RGB_32
	
#define YUV2RGB_32_NV21 \
	LOAD_UV_NV21 \
	YUV2RGB_32


void yuv420_rgb24_sse(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *U, const uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_PLANAR
			
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void yuv420_rgb24_sseu(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *U, const uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_PLANAR
			
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void nv12_rgb24_sse(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_NV12
			
			y_ptr1+=32;
			y_ptr2+=32;
			uv_ptr+=32;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void nv12_rgb24_sseu(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_NV12
			
			y_ptr1+=32;
			y_ptr2+=32;
			uv_ptr+=32;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void nv21_rgb24_sse(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_stream_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_NV21
			
			y_ptr1+=32;
			y_ptr2+=32;
			uv_ptr+=32;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}

void nv21_rgb24_sseu(
	uint32_t width, uint32_t height,
	const uint8_t *Y, const uint8_t *UV, uint32_t Y_stride, uint32_t UV_stride,
	uint8_t *RGB, uint32_t RGB_stride,
	YCbCrType yuv_type)
{
	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	const YUV2RGBParam *const param = &(YUV2RGB[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*uv_ptr=UV+(y/2)*UV_stride;
		
		uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		for(x=0; x<(width-31); x+=32)
		{
			YUV2RGB_32_NV21
			
			y_ptr1+=32;
			y_ptr2+=32;
			uv_ptr+=32;
			rgb_ptr1+=96;
			rgb_ptr2+=96;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
}



#endif //_YUVRGB_SSE2_


#ifdef _YUVRGB_AVX2_

#define AVX2_IMPLEMENT 3

#if AVX2_IMPLEMENT == 1

#define EXTRACT_MM256_TO_MM128(RS1, RS2, RS3, RS4, RS5, RS6, RS1L, RS1H, RS2L, RS2H, RS3L, RS3H, RS4L, RS4H, RS5L, RS5H, RS6L, RS6H) \
RS1L = _mm256_extractf128_si256(RS1, 0); \
RS1H = _mm256_extractf128_si256(RS1, 1); \
RS2L = _mm256_extractf128_si256(RS2, 0); \
RS2H = _mm256_extractf128_si256(RS2, 1); \
RS3L = _mm256_extractf128_si256(RS3, 0); \
RS3H = _mm256_extractf128_si256(RS3, 1); \
RS4L = _mm256_extractf128_si256(RS4, 0); \
RS4H = _mm256_extractf128_si256(RS4, 1); \
RS5L = _mm256_extractf128_si256(RS5, 0); \
RS5H = _mm256_extractf128_si256(RS5, 1); \
RS6L = _mm256_extractf128_si256(RS6, 0); \
RS6H = _mm256_extractf128_si256(RS6, 1);


#define UNPACK_RGB24_64_STEP1(RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8, RS9, RS10, RS11, RS12, RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8, RD9, RD10, RD11, RD12) \
RD1 = _mm_unpacklo_epi8(RS1, RS7); \
RD2 = _mm_unpackhi_epi8(RS1, RS7); \
RD3 = _mm_unpacklo_epi8(RS2, RS8); \
RD4 = _mm_unpackhi_epi8(RS2, RS8); \
RD5 = _mm_unpacklo_epi8(RS3, RS9); \
RD6 = _mm_unpackhi_epi8(RS3, RS9); \
RD7 = _mm_unpacklo_epi8(RS4, RS10); \
RD8 = _mm_unpackhi_epi8(RS4, RS10); \
RD9 = _mm_unpacklo_epi8(RS5, RS11); \
RD10 = _mm_unpackhi_epi8(RS5, RS11); \
RD11 = _mm_unpacklo_epi8(RS6, RS12); \
RD12 = _mm_unpackhi_epi8(RS6, RS12);


#define _mm256_set_m128i(v0, v1)  _mm256_insertf128_si256(_mm256_castsi128_si256(v0), (v1), 1)

#define COMBINE_MM128_TO_MM256(RS1L, RS1H, RS2L, RS2H, RS3L, RS3H, RS4L, RS4H, RS5L, RS5H, RS6L, RS6H, RS1, RS2, RS3, RS4, RS5, RS6) \
RS1 = _mm256_set_m128i(RS1L, RS1H); \
RS2 = _mm256_set_m128i(RS2L, RS2H); \
RS3 = _mm256_set_m128i(RS3L, RS3H); \
RS4 = _mm256_set_m128i(RS4L, RS4H); \
RS5 = _mm256_set_m128i(RS5L, RS5H); \
RS6 = _mm256_set_m128i(RS6L, RS6H);

#define UNPACK_RGB24_64_STEP1_POST(RS1, RS2, RS3, RS4, RS5, RS6) \
RS1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RS2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RS3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RS4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RS5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RS6 = _mm256_permute4x64_epi64 (RS6, 0xD8);


#define RGB2YUV_64 \
	__m256i r_32, g_32, b_32; \
	__m256i y1_32, y2_32, cb1_32, cb2_32, cr1_32, cr2_32, Y, cb, cr; \
	__m128i rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H; \
	__m128i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12; \
	__m256i rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+32)), \
		rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+64)), \
		rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2)), \
		rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+32)), \
		rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+64)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	EXTRACT_MM256_TO_MM128(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	UNPACK_RGB24_64_STEP1(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	UNPACK_RGB24_64_STEP1(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	COMBINE_MM128_TO_MM256(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP1_POST(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr1_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb1_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr1_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1+96)), \
	rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+128)), \
	rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+160)), \
	rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2+96)), \
	rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+128)), \
	rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+160)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	EXTRACT_MM256_TO_MM128(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	UNPACK_RGB24_64_STEP1(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	UNPACK_RGB24_64_STEP1(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H) \
	UNPACK_RGB24_64_STEP1(rgb1L, rgb1H, rgb2L, rgb2H, rgb3L, rgb3H, rgb4L, rgb4H, rgb5L, rgb5H, rgb6L, rgb6H, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12) \
	COMBINE_MM128_TO_MM256(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, tmp9, tmp10, tmp11, tmp12, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP1_POST(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr2_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1+32), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2+32), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb2_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr2_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm256_packus_epi16(cb1_32, cb2_32); \
	cr = _mm256_packus_epi16(cr1_32, cr2_32); \
	cb = _mm256_permute4x64_epi64 (cb, 0xD8); \
	cr = _mm256_permute4x64_epi64 (cr, 0xD8); \
	SAVE_SI256((__m256i*)(u_ptr), cb); \
	SAVE_SI256((__m256i*)(v_ptr), cr);

#elif AVX2_IMPLEMENT == 2

#define UNPACK_RGB24_64_STEP2A(RS1, RS2, RS3, RS4, RS5, RS6, RD1, RD2, RD3, RD4, RD5, RD6) \
RD1 = _mm256_unpacklo_epi8(RS1, RS4); \
RD2 = _mm256_unpackhi_epi8(RS1, RS4); \
RD3 = _mm256_unpacklo_epi8(RS2, RS5); \
RD4 = _mm256_unpackhi_epi8(RS2, RS5); \
RD5 = _mm256_unpacklo_epi8(RS3, RS6); \
RD6 = _mm256_unpackhi_epi8(RS3, RS6); \
RS1 = _mm256_permute2x128_si256(RD1, RD2, 0x20); \
RS2 = _mm256_permute2x128_si256(RD1, RD2, 0x31); \
RS3 = _mm256_permute2x128_si256(RD3, RD4, 0x20); \
RS4 = _mm256_permute2x128_si256(RD3, RD4, 0x31); \
RS5 = _mm256_permute2x128_si256(RD5, RD6, 0x20); \
RS6 = _mm256_permute2x128_si256(RD5, RD6, 0x31);

#define UNPACK_RGB24_64_STEP2A_POST(RS1, RS2, RS3, RS4, RS5, RS6) \
RS1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RS2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RS3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RS4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RS5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RS6 = _mm256_permute4x64_epi64 (RS6, 0xD8);


#define RGB2YUV_64 \
	__m256i r_32, g_32, b_32; \
	__m256i y1_32, y2_32, cb1_32, cb2_32, cr1_32, cr2_32, Y, cb, cr; \
	__m256i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6; \
	__m256i rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+32)), \
		rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+64)), \
		rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2)), \
		rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+32)), \
		rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+64)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A_POST(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr1_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb1_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr1_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1+96)), \
	rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+128)), \
	rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+160)), \
	rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2+96)), \
	rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+128)), \
	rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+160)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2A_POST(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr2_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1+32), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2+32), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb2_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr2_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm256_packus_epi16(cb1_32, cb2_32); \
	cr = _mm256_packus_epi16(cr1_32, cr2_32); \
	cb = _mm256_permute4x64_epi64 (cb, 0xD8); \
	cr = _mm256_permute4x64_epi64 (cr, 0xD8); \
	SAVE_SI256((__m256i*)(u_ptr), cb); \
	SAVE_SI256((__m256i*)(v_ptr), cr);

#elif AVX2_IMPLEMENT == 3

#define UNPACK_RGB24_64_STEP2B(RS1, RS2, RS3, RS4, RS5, RS6, RD1, RD2, RD3, RD4, RD5, RD6) \
RS1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RS2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RS3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RS4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RS5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RS6 = _mm256_permute4x64_epi64 (RS6, 0xD8); \
RD1 = _mm256_unpacklo_epi8(RS1, RS4); \
RD2 = _mm256_unpackhi_epi8(RS1, RS4); \
RD3 = _mm256_unpacklo_epi8(RS2, RS5); \
RD4 = _mm256_unpackhi_epi8(RS2, RS5); \
RD5 = _mm256_unpacklo_epi8(RS3, RS6); \
RD6 = _mm256_unpackhi_epi8(RS3, RS6);

#define UNPACK_RGB24_64_STEP2B_POST(RS1, RS2, RS3, RS4, RS5, RS6, RD1, RD2, RD3, RD4, RD5, RD6) \
RD1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RD2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RD3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RD4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RD5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RD6 = _mm256_permute4x64_epi64 (RS6, 0xD8);


#define RGB2YUV_64 \
	__m256i r_32, g_32, b_32; \
	__m256i y1_32, y2_32, cb1_32, cb2_32, cr1_32, cr2_32, Y, cb, cr; \
	__m256i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6; \
	__m256i rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1)), \
		rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+32)), \
		rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+64)), \
		rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2)), \
		rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+32)), \
		rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+64)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B_POST(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr1_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb1_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr1_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI256((const __m256i*)(rgb_ptr1+96)), \
	rgb2 = LOAD_SI256((const __m256i*)(rgb_ptr1+128)), \
	rgb3 = LOAD_SI256((const __m256i*)(rgb_ptr1+160)), \
	rgb4 = LOAD_SI256((const __m256i*)(rgb_ptr2+96)), \
	rgb5 = LOAD_SI256((const __m256i*)(rgb_ptr2+128)), \
	rgb6 = LOAD_SI256((const __m256i*)(rgb_ptr2+160)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	UNPACK_RGB24_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6) \
	UNPACK_RGB24_64_STEP2B_POST(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr2_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1+32), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb4, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2+32), Y); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb2_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr2_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm256_packus_epi16(cb1_32, cb2_32); \
	cr = _mm256_packus_epi16(cr1_32, cr2_32); \
	cb = _mm256_permute4x64_epi64 (cb, 0xD8); \
	cr = _mm256_permute4x64_epi64 (cr, 0xD8); \
	SAVE_SI256((__m256i*)(u_ptr), cb); \
	SAVE_SI256((__m256i*)(v_ptr), cr);

#endif


void rgb24_yuv420_avx2(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert(width % 32 == 0);
	assert(height % 2 == 0);

	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_store_si128
	#define LOAD_SI256 _mm256_load_si256
	#define SAVE_SI256 _mm256_store_si256
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);

	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;

		for(x=0; x<(width-63); x+=64)
		{
			RGB2YUV_64
			
			rgb_ptr1+=192;
			rgb_ptr2+=192;
			y_ptr1+=64;
			y_ptr2+=64;
			u_ptr+=32;
			v_ptr+=32;
		}

		if(x<(width-31))
		{
			RGB2YUV_32

			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
	#undef LOAD_SI256
	#undef SAVE_SI256
}

void rgb24_yuv420_avx2u(uint32_t width, uint32_t height,
	const uint8_t *RGB, uint32_t RGB_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert(width % 32 == 0);
	assert(height % 2 == 0);

	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	#define LOAD_SI256 _mm256_loadu_si256
	#define SAVE_SI256 _mm256_storeu_si256
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);

	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgb_ptr1=RGB+y*RGB_stride,
			*rgb_ptr2=RGB+(y+1)*RGB_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride;

		for(x=0; x<(width-63); x+=64)
		{
			RGB2YUV_64
			
			rgb_ptr1+=192;
			rgb_ptr2+=192;
			y_ptr1+=64;
			y_ptr2+=64;
			u_ptr+=32;
			v_ptr+=32;
		}

		if(x<(width-31))
		{
			RGB2YUV_32

			rgb_ptr1+=96;
			rgb_ptr2+=96;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
	#undef LOAD_SI256
	#undef SAVE_SI256
}


#define UNPACK_RGB32_64_STEP2B(RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8, RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8) \
RS1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RS2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RS3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RS4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RS5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RS6 = _mm256_permute4x64_epi64 (RS6, 0xD8); \
RS7 = _mm256_permute4x64_epi64 (RS7, 0xD8); \
RS8 = _mm256_permute4x64_epi64 (RS8, 0xD8); \
RD1 = _mm256_unpacklo_epi8(RS1, RS5); \
RD2 = _mm256_unpackhi_epi8(RS1, RS5); \
RD3 = _mm256_unpacklo_epi8(RS2, RS6); \
RD4 = _mm256_unpackhi_epi8(RS2, RS6); \
RD5 = _mm256_unpacklo_epi8(RS3, RS7); \
RD6 = _mm256_unpackhi_epi8(RS3, RS7); \
RD7 = _mm256_unpacklo_epi8(RS4, RS8); \
RD8 = _mm256_unpackhi_epi8(RS4, RS8);

#define UNPACK_RGB32_64_STEP2B_POST(RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8, RD1, RD2, RD3, RD4, RD5, RD6, RD7, RD8) \
RD1 = _mm256_permute4x64_epi64 (RS1, 0xD8); \
RD2 = _mm256_permute4x64_epi64 (RS2, 0xD8); \
RD3 = _mm256_permute4x64_epi64 (RS3, 0xD8); \
RD4 = _mm256_permute4x64_epi64 (RS4, 0xD8); \
RD5 = _mm256_permute4x64_epi64 (RS5, 0xD8); \
RD6 = _mm256_permute4x64_epi64 (RS6, 0xD8); \
RD7 = _mm256_permute4x64_epi64 (RS7, 0xD8); \
RD8 = _mm256_permute4x64_epi64 (RS8, 0xD8);


#define RGBA2YUVA_64 \
	__m256i r_32, g_32, b_32; \
	__m256i y1_32, y2_32, cb1_32, cb2_32, cr1_32, cr2_32, Y, cb, cr, A; \
	__m256i tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8; \
	__m256i rgb1 = LOAD_SI256((const __m256i*)(rgba_ptr1)), \
		rgb2 = LOAD_SI256((const __m256i*)(rgba_ptr1+32)), \
		rgb3 = LOAD_SI256((const __m256i*)(rgba_ptr1+64)), \
		rgb4 = LOAD_SI256((const __m256i*)(rgba_ptr1+96)), \
		rgb5 = LOAD_SI256((const __m256i*)(rgba_ptr2)), \
		rgb6 = LOAD_SI256((const __m256i*)(rgba_ptr2+32)), \
		rgb7 = LOAD_SI256((const __m256i*)(rgba_ptr2+64)), \
		rgb8 = LOAD_SI256((const __m256i*)(rgba_ptr2+96)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B_POST(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr1_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb7, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb7, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb1_32 = _mm256_add_epi16(cb1_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr1_32 = _mm256_add_epi16(cr1_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2), Y); \
	A = _mm256_unpacklo_epi8(rgb4, rgb8); \
	SAVE_SI256((__m256i*)(a_ptr1), A); \
	A = _mm256_unpackhi_epi8(rgb4, rgb8); \
	SAVE_SI256((__m256i*)(a_ptr2), A); \
	/* Rescale Cb and Cr to their final range */ \
	cb1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb1_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr1_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr1_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	\
	/* do the same again with next data */ \
	rgb1 = LOAD_SI256((const __m256i*)(rgba_ptr1+128)), \
	rgb2 = LOAD_SI256((const __m256i*)(rgba_ptr1+160)), \
	rgb3 = LOAD_SI256((const __m256i*)(rgba_ptr1+192)), \
	rgb4 = LOAD_SI256((const __m256i*)(rgba_ptr1+224)), \
	rgb5 = LOAD_SI256((const __m256i*)(rgba_ptr2+128)), \
	rgb6 = LOAD_SI256((const __m256i*)(rgba_ptr2+160)), \
	rgb7 = LOAD_SI256((const __m256i*)(rgba_ptr2+192)); \
	rgb8 = LOAD_SI256((const __m256i*)(rgba_ptr2+224)); \
	/* unpack rgb24 data to r, g and b data in separate channels*/ \
	/* see rgb.txt to get an idea of the algorithm, note that we only go to the next to last step*/ \
	/* here, because averaging in horizontal direction is easier like this*/ \
	/* The last step is applied further on the Y channel only*/ \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	UNPACK_RGB32_64_STEP2B(rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8) \
	UNPACK_RGB32_64_STEP2B_POST(tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7, tmp8, rgb1, rgb2, rgb3, rgb4, rgb5, rgb6, rgb7, rgb8) \
	/* first compute Y', (B-Y') and (R-Y'), in 16bits values, for the first line */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are saved*/ \
	r_32 = _mm256_unpacklo_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_sub_epi16(b_32, y1_32); \
	cr2_32 = _mm256_sub_epi16(r_32, y1_32); \
	r_32 = _mm256_unpacklo_epi8(rgb5, _mm256_setzero_si256()); \
	g_32 = _mm256_unpacklo_epi8(rgb6, _mm256_setzero_si256()); \
	b_32 = _mm256_unpacklo_epi8(rgb7, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr1+32), Y); \
	/* same for the second line, compute Y', (B-Y') and (R-Y'), in 16bits values */ \
	/* Y is saved for each pixel, while only sums of (B-Y') and (R-Y') for pairs of adjacents pixels are added to the previous values*/ \
	r_32 = _mm256_unpackhi_epi8(rgb1, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb2, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb3, _mm256_setzero_si256()); \
	y1_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y1_32 = _mm256_add_epi16(y1_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y1_32 = _mm256_srli_epi16(y1_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y1_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y1_32)); \
	r_32 = _mm256_unpackhi_epi8(rgb5, _mm256_setzero_si256()); \
	g_32 = _mm256_unpackhi_epi8(rgb6, _mm256_setzero_si256()); \
	b_32 = _mm256_unpackhi_epi8(rgb7, _mm256_setzero_si256()); \
	y2_32 = _mm256_add_epi16(_mm256_mullo_epi16(r_32, _mm256_set1_epi16(param->r_factor)), \
		_mm256_mullo_epi16(g_32, _mm256_set1_epi16(param->g_factor))); \
	y2_32 = _mm256_add_epi16(y2_32, _mm256_mullo_epi16(b_32, _mm256_set1_epi16(param->b_factor))); \
	y2_32 = _mm256_srli_epi16(y2_32, 8); \
	cb2_32 = _mm256_add_epi16(cb2_32, _mm256_sub_epi16(b_32, y2_32)); \
	cr2_32 = _mm256_add_epi16(cr2_32, _mm256_sub_epi16(r_32, y2_32)); \
	/* Rescale Y' to Y, pack it to 8bit values and save it */ \
	y1_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y1_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	y2_32 = _mm256_add_epi16(_mm256_srli_epi16(_mm256_mullo_epi16(y2_32, _mm256_set1_epi16(param->y_factor)), 7), _mm256_set1_epi16(param->y_offset)); \
	Y = _mm256_packus_epi16(y1_32, y2_32); \
	Y = _mm256_unpackhi_epi8(_mm256_slli_si256(Y, 8), Y); \
	SAVE_SI256((__m256i*)(y_ptr2+32), Y); \
	A = _mm256_unpacklo_epi8(rgb4, rgb8); \
	SAVE_SI256((__m256i*)(a_ptr1+32), A); \
	A = _mm256_unpackhi_epi8(rgb4, rgb8); \
	SAVE_SI256((__m256i*)(a_ptr2+32), A); \
	/* Rescale Cb and Cr to their final range */ \
	cb2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cb2_32, 2), _mm256_set1_epi16(param->cb_factor)), 8), _mm256_set1_epi16(128)); \
	cr2_32 = _mm256_add_epi16(_mm256_srai_epi16(_mm256_mullo_epi16(_mm256_srai_epi16(cr2_32, 2), _mm256_set1_epi16(param->cr_factor)), 8), _mm256_set1_epi16(128)); \
	/* Pack and save Cb Cr */ \
	cb = _mm256_packus_epi16(cb1_32, cb2_32); \
	cr = _mm256_packus_epi16(cr1_32, cr2_32); \
	cb = _mm256_permute4x64_epi64 (cb, 0xD8); \
	cr = _mm256_permute4x64_epi64 (cr, 0xD8); \
	SAVE_SI256((__m256i*)(u_ptr), cb); \
	SAVE_SI256((__m256i*)(v_ptr), cr);


void rgba32_yuva420_avx2(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint8_t *A, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert(width % 32 == 0);
	assert(height % 2 == 0);

	#define LOAD_SI128 _mm_load_si128
	#define SAVE_SI128 _mm_store_si128
	#define LOAD_SI256 _mm256_load_si256
	#define SAVE_SI256 _mm256_store_si256
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgba_ptr1=RGBA+y*RGBA_stride,
			*rgba_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride,
			*a_ptr1=A+y*Y_stride,
			*a_ptr2=A+(y+1)*Y_stride;
		
		for(x=0; x<(width-63); x+=64)
		{
			RGBA2YUVA_64
			
			rgba_ptr1+=256;
			rgba_ptr2+=256;
			y_ptr1+=64;
			y_ptr2+=64;
			u_ptr+=32;
			v_ptr+=32;
			a_ptr1+=64;
			a_ptr2+=64;
		}

		if(x<(width-31))
		{
			RGBA2YUVA_32

			rgba_ptr1+=128;
			rgba_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			a_ptr1+=32;
			a_ptr2+=32;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
	#undef LOAD_SI256
	#undef SAVE_SI256
}

void rgba32_yuva420_avx2u(
	uint32_t width, uint32_t height,
	const uint8_t *RGBA, uint32_t RGBA_stride,
	uint8_t *Y, uint8_t *U, uint8_t *V, uint8_t *A, uint32_t Y_stride, uint32_t UV_stride,
	YCbCrType yuv_type)
{
	assert(width % 32 == 0);
	assert(height % 2 == 0);

	#define LOAD_SI128 _mm_loadu_si128
	#define SAVE_SI128 _mm_storeu_si128
	#define LOAD_SI256 _mm256_loadu_si256
	#define SAVE_SI256 _mm256_storeu_si256
	const RGB2YUVParam *const param = &(RGB2YUV[yuv_type]);
	
	uint32_t x, y;
	for(y=0; y<(height-1); y+=2)
	{
		const uint8_t *rgba_ptr1=RGBA+y*RGBA_stride,
			*rgba_ptr2=RGBA+(y+1)*RGBA_stride;
		
		uint8_t *y_ptr1=Y+y*Y_stride,
			*y_ptr2=Y+(y+1)*Y_stride,
			*u_ptr=U+(y/2)*UV_stride,
			*v_ptr=V+(y/2)*UV_stride,
			*a_ptr1=A+y*Y_stride,
			*a_ptr2=A+(y+1)*Y_stride;
		
		for(x=0; x<(width-63); x+=64)
		{
			RGBA2YUVA_64
			
			rgba_ptr1+=256;
			rgba_ptr2+=256;
			y_ptr1+=64;
			y_ptr2+=64;
			u_ptr+=32;
			v_ptr+=32;
			a_ptr1+=64;
			a_ptr2+=64;
		}

		if(x<(width-31))
		{
			RGBA2YUVA_32

			rgba_ptr1+=128;
			rgba_ptr2+=128;
			y_ptr1+=32;
			y_ptr2+=32;
			u_ptr+=16;
			v_ptr+=16;
			a_ptr1+=32;
			a_ptr2+=32;
		}
	}
	#undef LOAD_SI128
	#undef SAVE_SI128
	#undef LOAD_SI256
	#undef SAVE_SI256
}

#endif // _YUVRGB_AVX2_
