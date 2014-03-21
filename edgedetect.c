#include "header.h"
#include <math.h>
#include <stdlib.h>

float * array_create(const int* size)
{
	return (float*)malloc(*size);
}

void array_free(float *ptr)
{
	if (ptr) 
		free(ptr);
}

void convert_grayscale(float* r, float* g, float* b, float* gray, const int* size)
{
	int position;
	#pragma omp for
	for (position= 0; position < (*size); position++)
		gray[position] = (r[position] + g[position] + b[position]) / 3.f;
}

float convolve(float* kernel, float* ringbuf, const int* ksize, int i0)
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

void gaussian_blur(float* src, float* dst, const int* width, const int* height, float sigma)
{
	const int ksize = (int)(sigma * 2.f * 4.f + 1) | 1;
	const int ksizeColumnLimit=ksize-1;
	const int halfkColumn = ksize / 2;
	const int halfkRow=halfkColumn*(*width);
	const int xmax = (*width) - halfkColumn;
	const int ymax = (*height)-halfkRow;
	const int maxColumnLimit=(*width)-1;
	const int maxRowLimit=(*height)-(*width);
	float scale = -0.5f/(sigma*sigma);
	float sum = 0.f,tmp,t;
	float *kernel, *ringbuf;
	int x, y, i,position,bufi0;
	// if sigma too small, just copy src to dst
	if (ksize <= 1)
	{
		for (y = 0; y < (*height); y++)
			dst[y] = src[y];
		return;
	}

	// create Gaussian kernel
	kernel = malloc(ksize * sizeof(float));

	for (i = 0; i < ksize; i++)
	{
		tmp = (float)(i - halfkColumn);
		t = expf(scale * tmp * tmp);
		kernel[i] = t;
		sum += t;
	}

	scale = 1.f / sum;
	for (i = 0; i < ksize; i++)
		kernel[i] *= scale;
	#pragma omp parallel private(x,y,ringbuf,tmp,bufi0,position)
	// blur each row
	{
		ringbuf = malloc(ksize * sizeof(float));
		#pragma omp for
		for (y = 0; y < (*height); y+=(*width))
		{
			bufi0 = ksizeColumnLimit;
			tmp = src[y];
			for (x = 0; x < halfkColumn  ; x++) ringbuf[x] = tmp;
			for ( position=y; x < ksizeColumnLimit; x++) ringbuf[x] = src[position++];
			position=y;
			for (x = 0; x < xmax; x++)
			{
				ringbuf[bufi0++] = src[position+halfkColumn];
				if (bufi0 == ksize) bufi0 = 0;
				dst[position++] = convolve(kernel, ringbuf, &ksize, bufi0);
			}

			for (tmp = src[y+maxColumnLimit] ; x < (*width); x++)
			{
				ringbuf[bufi0++] = tmp;
				if (bufi0 == ksize) bufi0 = 0;
				dst[position++] = convolve(kernel, ringbuf, &ksize, bufi0);
			}
		}

		// blur each column
		#pragma omp for
		for (x = 0; x < (*width); x++)
		{
			bufi0 =ksizeColumnLimit;
			tmp = dst[x];
			for (y = 0; y < halfkColumn  ; y++) ringbuf[y] = tmp;
			for (position=x; y < ksizeColumnLimit; y++)
			{
				ringbuf[y] = dst[position];
				position+=(*width);
			}
			position=x;
			for (y = 0; y < ymax; y+=(*width))
			{
				ringbuf[bufi0++] = dst[position+halfkRow];
				if (bufi0 == ksize) bufi0 = 0;
				dst[position] = convolve(kernel, ringbuf, &ksize, bufi0);
				position+=(*width);
			}

			for (tmp = dst[maxRowLimit+x]; y < (*height); y+=(*width))
			{
				ringbuf[bufi0++] = tmp;
				if (bufi0 == ksize) bufi0 = 0;
				dst[position] = convolve(kernel, ringbuf, &ksize, bufi0);
				position+=(*width);
			}
		}
		if(ringbuf)
			free(ringbuf);
	}
	// clean up
	free(kernel);
}

void compute_gradient(float* src, const int* width, const int* height, float* g_mag, float* g_ang)
{
	// Sobel mask values
	const float mx[]={-0.25f, 0.f, 0.25f,-0.5f , 0.f, 0.5f,-0.25f, 0.f, 0.25f};
	const float my[]={-0.25f,-0.5f,-0.25f,0.f  , 0.f , 0.f,0.25f, 0.5f, 0.25f};
	const int maxRowLimit=(*height)-(*width);
	const int maxColumnLimit=(*width)-1;
	int y, x, i, j, r, c,mPosition,srcRow,srcColumn;
	float gx,gy;
	#pragma omp parallel for private(y,x,i,j,r,c,mPosition,srcRow,srcColumn,gx,gy)
	for (y = (*width); y < maxRowLimit; y+=(*width))
	for (x = 0; x < (*width); x++)
	{
		gx = 0.f, gy = 0.f;
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

int is_edge(float* g_mag, float* g_ang, float* threshold, const int* x, const int* y, const int* width, const int* height)
{
	int position=(*y)+(*x);
	if (g_mag[position] >= (*threshold))
	{
		int dir = ((int) roundf(g_ang[position]/M_PI_4) + 4) % 4;
		// horizontal gradient : vertical edge
		if (dir == 0)
		{
			float left  = g_mag[position- ((*x)>0  )];
			float right = g_mag[position+ ((*x)<(*width)-1)];
			return (g_mag[position] >= left && g_mag[position] >= right);
		}
		int belowLimit=(((*y)<(*height)-(*width))?(*width)+position:position);
		int aboveLimit=(((*y)>0  )?position-(*width):position);
		// vertical gradient : horizontal edge
		if (dir == 2)
		{
			float above = g_mag[aboveLimit];
			float below = g_mag[belowLimit];
			return(g_mag[position] >= above && g_mag[position] >= below);
		}
		// diagonal gradient : diagonal edge
		if (dir == 1)
		{
			float above_l = g_mag[aboveLimit- ((*x)>0  )];
			float below_r = g_mag[belowLimit+ ((*x)<(*width)-1)];
			return(g_mag[position] >= above_l && g_mag[position] >= below_r);
		}
		// diagonal gradient : diagonal edge
		if (dir == 3)
		{
			float above_r = g_mag[aboveLimit+ ((*x)<(*width)-1)];
			float below_l = g_mag[belowLimit- ((*x)>0  )];
			return(g_mag[position] >= above_r && g_mag[position] >= below_l);
		}
	}
	return 0;
}

void detect_edges(Image* img, float sigma, float threshold, unsigned char* edge_pix, PList *edge_pts)
{
	const int width = img->w;
	const int height = (img->h)*width;
	const int size=height*sizeof(float);
	const int maxRowLimit=height-width;
	const int maxColumnLimit=width-1;
	int x,y;
	// convert image to grayscale
	float* gray = array_create(&size);
	convert_grayscale(img->r,img->g,img->b,gray,&height);

	// blur grayscale image
	float* gray2 = array_create(&size);
	gaussian_blur(gray, gray2, &width, &height, sigma);

	// compute gradient of blurred image
	float* g_mag = array_create(&size);
	float* g_ang = array_create(&size);
	compute_gradient(gray2, &width, &height, g_mag, g_ang);

	// mark edge pixels
	#define PIX(y,x) edge_pix[(y)+(x)]
	#pragma omp parallel private(x,y)
	{
		#pragma omp for
		for (y = 0; y < height; y+=width)
		for (x = 0; x < width; x++)
		{
			PIX(y,x) = is_edge(g_mag,g_ang,&threshold,&x,&y,&width,&height) ? 255 : 0;
		}
		
		// connect horizontal edges
		#pragma omp for
		for (y = 0; y < height; y+=width)
		for (x = 1; x <maxColumnLimit; x++)
		{
			if (!PIX(y,x) && PIX(y,x+1) && PIX(y,x-1))
				PIX(y,x) = 255;
		}
		
		// connect vertical edges
		#pragma omp for
		for (x = 0; x < width  ; x++)
		for (y = width; y < maxRowLimit; y+=width)
		{
			if (!PIX(y,x) && PIX(y+width,x) && PIX(y-width,x))
				PIX(y,x) = 255;
		}

		// connect diagonal edges
		#pragma omp for
		for (y = width; y <maxRowLimit; y+=width)
		for (x = 1; x < maxColumnLimit; x++)
		{
			if (!PIX(y,x) && PIX(y-width,x-1) && PIX(y+width,x+1))
				PIX(y,x) = 255;
			if (!PIX(y,x) && PIX(y-width,x+1) && PIX(y+width,x-1))
				PIX(y,x) = 255;
		}
	}
	// add edge points to list
	if (edge_pts)
	{
		int temp=0;
		for (y = 0; y < height; y+=width)
		{
			for (x = 0; x < width; x++)
			{
				if (PIX(y,x)) PList_push(edge_pts, x, temp);
			}
			temp++;
		}
	}

	// cleanup
	array_free(g_mag);
	array_free(g_ang);
	array_free(gray2);
	array_free(gray);
}
