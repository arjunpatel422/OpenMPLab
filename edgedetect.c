#include "header.h"
#include <math.h>
#include <stdlib.h>

float * array_create(const int* size)
{
	float * ptr = malloc((*size));
	return ptr;
}

void array_free(float *ptr)
{
	if (ptr) 
		free(ptr);
}

void convert_grayscale(Image *img, float *gray)
{
	int position;
	const int size=img->w*img->h;
	for (position= 0; position < size; position++)
	{
		float r = img->r[position];
		float g = img->g[position];
		float b = img->b[position];
		gray[position] = (r + g + b) / 3.f;
	}
}

float convolve(float *kernel, float *ringbuf, const int* ksize, int i0)
{
	int i;
	float sum = 0.f;
	for (i = 0; i < (*ksize); i++)
	{
		sum += kernel[i] * ringbuf[i0++];
		if (i0 == (*ksize)) i0 = 0;
	}
	return sum;
}

void gaussian_blur(float *src, float *dst, const int* width, const int* height, float sigma)
{
	int x, y, i,position;
	const int ksize = (int)(sigma * 2.f * 4.f + 1) | 1;
	const int ksizeColumnLimit=ksize-1;
	const int halfkColumn = ksize / 2;
	const int ksizeRowLimit=ksizeColumnLimit*(*width);
	const int halfkRow=halfkColumn*(*width);
	const int xmax = (*width) - halfkColumn;
	const int ymax = (*height)-halfkRow;
	const int maxColumnLimit=(*width)-1;
	const int maxRowLimit=(*height)-(*width);
	float scale = -0.5f/(sigma*sigma);
	float sum = 0.f,s,t;
	float *kernel, *ringbuf;

	// if sigma too small, just copy src to dst
	if (ksize <= 1)
	{
		for (y = 0; y < (*height); y++)
			dst[y] = src[y];
		return;
	}

	// create Gaussian kernel
	kernel = malloc(ksize * sizeof(float));
	ringbuf = malloc(ksize * sizeof(float));

	for (i = 0; i < ksize; i++)
	{
		float x = (float)(i - halfkColumn);
		float t = expf(scale * x * x);
		kernel[i] = t;
		sum += t;
	}

	scale = 1.f / sum;
	for (i = 0; i < ksize; i++)
		kernel[i] *= scale;

	// blur each row
	for (y = 0; y < (*height); y+=(*width))
	{
		int bufi0 = ksizeColumnLimit;
		float tmp = src[y];
		for (x = 0; x < halfkColumn  ; x++) ringbuf[x] = tmp;
		for (     ; x < ksizeColumnLimit; x++) ringbuf[x] = src[y+x-halfkColumn];

		for (x = 0; x < xmax; x++)
		{
			ringbuf[bufi0++] = src[y+x+halfkColumn];
			if (bufi0 == ksize) bufi0 = 0;
			dst[y+x] = convolve(kernel, ringbuf, &ksize, bufi0);
		}

		tmp = src[y+maxColumnLimit];
		for ( ; x < (*width); x++)
		{
			ringbuf[bufi0++] = tmp;
			if (bufi0 == ksize) bufi0 = 0;
			dst[y+x] = convolve(kernel, ringbuf, &ksize, bufi0);
		}
	}

	// blur each column
	for (x = 0; x < (*width); x++)
	{
		int bufi0 =ksizeColumnLimit;
		float tmp = dst[x];
		for (y = 0; y < halfkColumn  ; y++) ringbuf[y] = tmp;
		for (position=x; y < ksizeColumnLimit; y++)
		{
			ringbuf[y] = dst[position];
			position+=(*width);
		}

		for (y = 0; y < ymax; y+=(*width))
		{
			ringbuf[bufi0++] = dst[y+halfkRow+x];
			if (bufi0 == ksize) bufi0 = 0;

			dst[y+x] = convolve(kernel, ringbuf, &ksize, bufi0);
		}

		tmp = dst[maxRowLimit+x];
		for ( ; y < (*height); y+=(*width))
		{
			ringbuf[bufi0++] = tmp;
			if (bufi0 == ksize) bufi0 = 0;

			dst[y+x] = convolve(kernel, ringbuf, &ksize, bufi0);
		}
	}

	// clean up
	free(kernel);
	free(ringbuf);
}

void compute_gradient(float *src, const int* width, const int* height, float *g_mag, float *g_ang)
{
	// Sobel mask values
	const float mx[]={-0.25f, 0.f, 0.25f,-0.5f , 0.f, 0.5f,-0.25f, 0.f, 0.25f};
	const float my[]={-0.25f,-0.5f,-0.25f,0.f  , 0.f , 0.f,0.25f, 0.5f, 0.25f};
	const int maxRowLimit=(*height)-(*width);
	const int maxColumnLimit=(*width)-1;
	int y, x, i, j, r, c,mPosition,srcRow,srcColumn;
	for (y = 0; y < (*height); y+=(*width))
	for (x = 0; x < (*width); x++)
	{
		float gx = 0.f, gy = 0.f;
		mPosition=0;
		srcRow=y-(*width);
		for (i = 0; i < 3; i++)
		{
			srcColumn=x-1;
			for (j = 0; j < 3; j++)
			{
				r = srcRow; if(r<0)r=0; else if(r>maxRowLimit)r=maxRowLimit;
				c = srcColumn; if(c<0)c=0; else if(c>maxColumnLimit)c=maxColumnLimit;			
				gx += src[r+c] * mx[mPosition];
				gy += src[r+c] * my[mPosition];
				mPosition++;
				srcColumn++;
			}
			srcRow+=(*width);
		}
		g_mag[y+x] = hypotf(gy, gx);
		g_ang[y+x] = atan2f(gy, gx);
	}
}

int is_edge(float *g_mag, float *g_ang, float threshold, int x, int y, const int* width, const int* height)
{
	if (g_mag[y+x] >= threshold)
	{
		int dir = ((int) roundf(g_ang[y+x]/M_PI_4) + 4) % 4;

		// horizontal gradient : vertical edge
		if (dir == 0)
		{
			float left  = g_mag[y+x - (x>0  )];
			float right = g_mag[y+x + (x<(*width)-1)];
			return (g_mag[y+x] >= left && g_mag[y+x] >= right);
		}
		// vertical gradient : horizontal edge
		else if (dir == 2)
		{
			float above = g_mag[y - ((y>0  )*(*width))+x];
			float below = g_mag[y + ((y<(*height)-(*width))*(*width))+x];
			return (g_mag[y+x] >= above && g_mag[y+x] >= below);
		}
		// diagonal gradient : diagonal edge
		else if (dir == 1)
		{
			float above_l = g_mag[y - ((y>0  )*(*width))+x - (x>0  )];
			float below_r = g_mag[y + ((y<(*height)-(*width))*(*width))+x + (x<(*width)-1)];
			return (g_mag[y+x] >= above_l && g_mag[y+x] >= below_r);
		}
		// diagonal gradient : diagonal edge
		else if (dir == 3)
		{
			float above_r = g_mag[y - (((y>0  ))*(*width))+x + (x<(*width)-1)];
			float below_l = g_mag[y + (((y<(*height)-1))*(*width))+x - (x>0  )];
			return (g_mag[y+x] >= above_r && g_mag[y+x] >= below_l);
		}
	}
	return 0;
}

void detect_edges(Image *img, float sigma, float threshold, unsigned char *edge_pix, PList *edge_pts)
{
	int x, y;
	const int width = img->w;
	const int height = (img->h)*width;
	const int size=height*sizeof(float);
	const int maxRowLimit=height-width;
	const int maxColumnLimit=width-1;
	// convert image to grayscale
	float *gray = array_create(&size);
	convert_grayscale(img, gray);

	// blur grayscale image
	float *gray2 = array_create(&size);
	gaussian_blur(gray, gray2, &width, &height, sigma);

	// compute gradient of blurred image
	float *g_mag = array_create(&size);
	float *g_ang = array_create(&size);
	compute_gradient(gray2, &width, &height, g_mag, g_ang);

	// mark edge pixels
	#define PIX(y,x) edge_pix[(y)+(x)]
	for (y = 0; y < height; y+=width)
	for (x = 0; x < width; x++)
	{
		PIX(y,x) = is_edge(g_mag,g_ang,threshold,x,y,&width,&height) ? 255 : 0;
	}

	// connect horizontal edges
	for (y = 0; y < height; y+=width)
	for (x = 1; x <maxColumnLimit; x++)
	{
		if (!PIX(y,x) && PIX(y,x+1) && PIX(y,x-1))
			PIX(y,x) = 255;
	}

	// connect vertical edges
	for (x = 0; x < width  ; x++)
	for (y = 1; y < maxRowLimit; y+=width)
	{
		if (!PIX(y,x) && PIX(y+width,x) && PIX(y-width,x))
			PIX(y,x) = 255;
	}

	// connect diagonal edges
	for (y = 1; y <maxRowLimit; y+=width)
	for (x = 1; x < maxColumnLimit; x++)
	{
		if (!PIX(y,x) && PIX(y-width,x-1) && PIX(y+width,x+1))
			PIX(y,x) = 255;
		if (!PIX(y,x) && PIX(y-width,x+1) && PIX(y+width,x-1))
			PIX(y,x) = 255;
	}

	// add edge points to list
	if (edge_pts)
	{
		for (y = 0; y < height; y+=width)
		for (x = 0; x < width; x++)
		{
			if (PIX(y,x)) PList_push(edge_pts, x, y/width);
		}
	}

	// cleanup
	array_free(g_mag);
	array_free(g_ang);
	array_free(gray2);
	array_free(gray);
}
