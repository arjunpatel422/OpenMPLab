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
	const int ksizePlus1=ksize+1;
	const int ksizeLimit=ksize-1;
	const int ksizeMinus2=ksizeLimit-1;
	const int halfkColumn = ksize / 2;
	const int halfkColumnPlus1=halfkColumn+1;
	const int halfkRow=halfkColumn*(*width);
	const int halfkRowPlusWidth=halfkRow+(*width);
	const int xmax = (*width) - halfkColumn;
	const int ymax = (*height)-halfkRow;
	const int maxColumnLimit=(*width)-1;
	const int ringbufSize=(*height)/(*width);
	const int ringbufSizeLimit=ringbufSize-1;
	const int ringbufMax=ringbufSize-ksizeLimit;
	float scale = -0.5f/(sigma*sigma);
	float sum = 0.f,tmp,t;
	float *kernel;
	int y;
	// if sigma too small, just copy src to dst
	if (ksize <= 1)
	{
		for (y = 0; y < (*height); y++)
			dst[y] = src[y];
		return;
	}

	// create Gaussian kernel
	kernel = malloc(ksize * sizeof(float));

	for (y = 0; y < ksize; y++)
	{
		tmp = (float)(y - halfkColumn);
		t = expf(scale * tmp * tmp);
		kernel[y] = t;
		sum += t;
	}

	scale = 1.f / sum;
	for (y = 0; y < ksize; y++)
		kernel[y] *= scale;
	#pragma omp parallel private (y,sum,tmp)
	{
		int x,row,column,temp,position,fixedy;
		float sum2=1.f;
		// blur each row
		#pragma omp for
		for (row = 0; row < (*height); row+=(*width))
		{
			position=row;
			tmp = src[row];
			fixedy=position+1;
			for (column = 0; column < halfkColumnPlus1; column++ )
				{
					x=0;
					y =fixedy;
					sum = 0.f;
					for (  ; x < (halfkColumnPlus1- column) ; x++)  
						sum += kernel[x];	//adds the first halfk plus 1 elements of kernel to the sum.   
					sum *= tmp;  //adds up the first halfk plus 1 elements of ringbuf/source terms to the sum. 
					for (  ; x < ksizeLimit; x++, y++)
						sum += kernel[x] * src[y]; 
					dst[position++] = sum + (kernel[x] * src[y]);	
				}
								
			for (; column < xmax ; column++ )
			{
				y =fixedy;
				fixedy++;
				sum = 0.f;
				for (x=0  ; x < (halfkColumnPlus1 - column) ; x++)   
					sum = sum + (kernel[x] * tmp);
				for (  ; x < ksizeLimit ; x++, y++)
					sum += kernel[x] * src[y]; 
				dst[position++] = sum + (kernel[x] * src[y]);
			}				
			tmp = src[row+maxColumnLimit];
			temp=ksizeMinus2;
			for (fixedy=position-halfkColumn; column < (*width) ; column++, temp--)
			{
				y =fixedy;
				fixedy++;
				sum = 0.f;
				for (x = 0; x < temp ; x++, y++)
					sum += kernel[x] * src[y]; 
				for (sum2 = 0.f ; x < ksizePlus1 ; x++)
					sum2 += kernel[x];
				sum += (sum2 * tmp);		
				dst[position++] = sum;
			}
		}
		float * bigRingbuf = malloc(ringbufSize*sizeof(float));	
		// blur each column
		#pragma omp for
		for (column = 0; column < (*width); column++)
		{
			tmp  = dst[column];
			for (fixedy=column,y = 0 ; y <ringbufSize ; y++,fixedy+=(*width))
				bigRingbuf[y] = dst[fixedy];
			position=column;
			for (fixedy=0,row = 0 ; row < halfkRowPlusWidth ; row+=(*width))
			{
				sum = 0.f;		
				for (x = 0; x < (halfkColumnPlus1-fixedy) ; x++)
					sum += kernel[x];
				y =1;	
				for (sum *= tmp; x < ksizeLimit; x++, y++)
					sum+=(kernel[x] * bigRingbuf[y]); 	
				dst[position] = sum + (kernel[x] * bigRingbuf[y]);
				position+=(*width);
				fixedy++;
			}
			for (fixedy=1; row < ymax ; row+=(*width))
			{
				sum = 0.f;
				for (x = 0  ; x < (halfkColumnPlus1 - row/(*width)) ; x++)
					sum+=(kernel[x] * tmp);	
				for (y=fixedy; x < ksizeLimit; x++, y++)
					sum+=(kernel[x] * bigRingbuf[y]); 	
				dst[position] = sum + (kernel[x] * bigRingbuf[y]);
				position+=(*width);
				fixedy++;
			}
			tmp= bigRingbuf[ringbufSizeLimit];
			temp=ksizeMinus2;
			fixedy=ringbufMax;
			for (; row < (*height) ; row+=(*width), temp--)
			{		
				sum = 0.f;
				y =fixedy;
				for (x = 0 ; x < temp ; x++, y++)
					sum += kernel[x] * bigRingbuf[y]; 
				for (sum2 = 0.f ; x < ksizePlus1; x++)
					sum2 += kernel[x];
				sum += (sum2 * tmp);
				dst[position] = sum;
				position+=(*width);
				fixedy++;
			}
		}
		if(bigRingbuf)
			free(bigRingbuf);
	}
	// clean up
	free(kernel);
}

void compute_gradient(float* src, const int* width, const int* height, float* g_mag, float* g_ang)
{
	const float mx[]={-0.25f, 0.f, 0.25f,-0.5f , 0.f, 0.5f,-0.25f, 0.f, 0.25f};
	const float my[]={-0.25f,-0.5f,-0.25f,0.f  , 0.f , 0.f, 0.25f, 0.5f, 0.25f};
	const int maxRowLimit=(*height)-(*width);
	const int maxColumnLimit=(*width)-1;
	const int doubleWidth=((*width)+(*width));
	int y, x, srcPosition,mPosition;
	float gx,gy;

	#pragma omp parallel for private(y,x,srcPosition,mPosition,gx,gy)
	for (y =(*width); y <maxRowLimit; y+=(*width))
	{
		//Covers West Edge
		gx = 0.f;
		gy = 0.f;
		srcPosition=y-(*width);
		for (mPosition = 0; mPosition < 9; srcPosition+=(*width))
		{
			gx += src[srcPosition] * (mx[mPosition]+mx[mPosition+1]);
			gy += src[srcPosition++] * (my[mPosition]+my[mPosition+1]);
			mPosition++;
			gx+=src[srcPosition]*mx[++mPosition];
			gy+=src[srcPosition--]*my[mPosition++];
		}
		srcPosition=y;
		g_mag[srcPosition] = hypotf(gy, gx);
		g_ang[srcPosition] = atan2f(gy, gx);
		//End of Covering West Edge
		srcPosition++;
		for (x = 1; x <maxColumnLimit; x++)
		{
			gx = 0.f;
			gy = 0.f;
			srcPosition-=(*width);
			for (mPosition=0; mPosition< 9; srcPosition+=(*width))
			{
				gx += src[--srcPosition] * mx[mPosition];
				gy += src[srcPosition++] * my[mPosition++];
				gx += src[srcPosition] * mx[mPosition];
				gy += src[srcPosition++] * my[mPosition++];
				gx += src[srcPosition] * mx[mPosition];
				gy += src[srcPosition--] * my[mPosition++];
			}
			srcPosition-=doubleWidth;
			g_mag[srcPosition] = hypotf(gy, gx);
			g_ang[srcPosition] = atan2f(gy, gx);
			srcPosition++;
		}
		//Covers East Edge
		gx = 0.f;
		gy = 0.f;
		srcPosition-=(*width);
		for (mPosition=0; mPosition < 9; srcPosition+=(*width))
		{
			gx+=src[--srcPosition]*mx[mPosition];
			gy+=src[srcPosition++]*my[mPosition++];
			gx+=src[srcPosition]*(mx[mPosition]+mx[mPosition+1]);
			gy+=src[srcPosition]*(my[mPosition]+my[mPosition+1]);
			mPosition+=2;
		}
		srcPosition-=doubleWidth;
		g_mag[srcPosition] = hypotf(gy, gx);
		g_ang[srcPosition] = atan2f(gy, gx);
		//Finishes covering east edge
	}
}

int is_edge(float* g_mag, float* g_ang, float* threshold, const int* x, const int* y, const int* width, const int* maxColumnLimit,const int* maxRowLimit)
{
	int position=(*y)+(*x);
	if (g_mag[position] >= (*threshold))
	{
		int dir = ((int) roundf(g_ang[position]/M_PI_4) + 4) % 4;
		// horizontal gradient : vertical edge
		if (!dir)
		{
			float left  = g_mag[position- ((*x)>0  )];
			float right = g_mag[position+ ((*x)<(*maxColumnLimit))];
			return (g_mag[position] >= left && g_mag[position] >= right);
		}
		int belowLimit=(((*y)<(*maxRowLimit))?(*width)+position:position);
		int aboveLimit=(((*y)>0  )?position-(*width):position);
		// vertical gradient : horizontal edge
		if (dir == 2)
		{
			float above = g_mag[aboveLimit];
			float below = g_mag[belowLimit];
			return(g_mag[position] >= above && g_mag[position] >= below);
		}
		// diagonal gradient : diagonal edge
		else if (dir == 1)
		{
			float above_l = g_mag[aboveLimit- ((*x)>0  )];
			float below_r = g_mag[belowLimit+ ((*x)<(*maxColumnLimit))];
			return(g_mag[position] >= above_l && g_mag[position] >= below_r);
		}
		// diagonal gradient : diagonal edge
		else if (dir == 3)
		{
			float above_r = g_mag[aboveLimit+ ((*x)<(*maxColumnLimit))];
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
			PIX(y,x) = is_edge(g_mag,g_ang,&threshold,&x,&y,&width,&maxColumnLimit,&maxRowLimit) ? 255 : 0;
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
