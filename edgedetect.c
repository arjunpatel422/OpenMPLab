#include "header.h"
#include <math.h>
#include <stdlib.h>

float * array_create(const int* size) //This function is just kept for code clarity
{
	return (float*)malloc(*size); //This just casts it as a safety
}

void array_free(float *ptr) //This code stops redundancy
{
	if (ptr) //This frees the dynamic memory
		free(ptr);
}

void convert_grayscale(float* r, float* g, float* b, float* gray, const int* size)
{
	int position;
	#pragma omp for//This is parallelized because there are a lot of iterations and barely any code to do
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
	const int ringbufSize=(*height)/(*width); //These were created so ringbuf wouldn't have to be ridiculously large
	const int ringbufArraySize=ringbufSize*sizeof(float);//This is to reduce multiplication
	const int ringbufSizeLimit=ringbufSize-1; //This allows instant access to the end of the pointer
	const int ringbufMax=ringbufSize-ksizeLimit;//This is for simplicity
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
		int x,row,column,temp,position,fixedy;//The fixedy variable is designed to act as a setter for the y variable and increments automatically
		float sum2=1.f;	//The position variable is done to always point to where the work is and be independent of other variables as much as possible
		// blur each row
		#pragma omp for
		for (row = 0; row < (*height); row+=(*width))
		{
			position=row;
			tmp = src[row];
			fixedy=position+1;
			for (column = 0; column < halfkColumnPlus1; column++ )
				{
					sum = 0.f;
					for (x=0  ; x < (halfkColumnPlus1- column) ; x++)  
						sum += kernel[x];	//adds the first halfk plus 1 elements of kernel to the sum.   
					sum *= tmp;  //adds up the first halfk plus 1 elements of ringbuf/source terms to the sum. 
					for (y =fixedy  ; x < ksizeLimit; x++, y++)
						sum += kernel[x] * src[y]; 
					dst[position++] = sum + (kernel[x] * src[y]); //I increment the position repeatedly for every column	
				}
								
			for (; column < xmax ; column++ )
			{
				sum = 0.f;
				for (x=0  ; x < (halfkColumnPlus1 - column) ; x++)   
					sum+=(kernel[x] * tmp);
				y =fixedy;//This resets y to what it should be
				fixedy++;//I do this here for better efficiency while it is current in memory
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
		float * ringbuf = malloc(ringbufArraySize);	
		// blur each column
		#pragma omp for
		for (column = 0; column < (*width); column++)
		{
			tmp  = dst[column];
			for (fixedy=column,y = 0 ; y <ringbufSize ; y++,fixedy+=(*width))
				ringbuf[y] = dst[fixedy];
			for (position=column,row = 0 ; row < halfkRowPlusWidth ; row+=(*width))
			{
				sum = 0.f;		
				for (x = 0; x < (halfkColumnPlus1-fixedy) ; x++)
					sum += kernel[x];
				y =1;	
				for (sum *= tmp; x < ksizeLimit; x++, y++)
					sum+=(kernel[x] * ringbuf[y]); 	
				dst[position] = sum + (kernel[x] * ringbuf[y]);
				position+=(*width);
			}
			for (fixedy=1; row < ymax ; row+=(*width))
			{
				sum = 0.f;
				for (x = 0  ; x < (halfkColumnPlus1-fixedy) ; x++)
					sum+=(kernel[x] * tmp);	
				y=fixedy;
				for (fixedy++; x < ksizeLimit; x++, y++)
					sum+=(kernel[x] * ringbuf[y]); 	
				dst[position] = sum + (kernel[x] * ringbuf[y]);
				position+=(*width);
			}
			tmp= ringbuf[ringbufSizeLimit];
			temp=ksizeMinus2;
			for (fixedy=ringbufMax; row < (*height) ; row+=(*width), temp--)
			{		
				y =fixedy;
				fixedy++;
				sum = 0.f;
				for (x = 0 ; x < temp ; x++, y++)
					sum += kernel[x] * ringbuf[y]; 
				for (sum2 = 0.f ; x < ksizePlus1; x++)
					sum2 += kernel[x];
				sum += (sum2 * tmp);
				dst[position] = sum;
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
	const float mx[]={-0.25f, 0.f, 0.25f,-0.5f , 0.f, 0.5f,-0.25f, 0.f, 0.25f}; //This is for better efficiency for multiplication
	const float my[]={-0.25f,-0.5f,-0.25f,0.f  , 0.f , 0.f, 0.25f, 0.5f, 0.25f};//This is for the same reason. 
	const int maxRowLimit=(*height)-(*width);
	const int maxColumnLimit=(*width)-1;
	const int doubleWidth=((*width)+(*width));
	int y, x, srcPosition,mPosition;
	float gx,gy;
	//The main idea for what I do is to make all calculations independent of other variables for fewer operations
	#pragma omp parallel for private(y,x,srcPosition,mPosition,gx,gy)
	for (y =(*width); y <maxRowLimit; y+=(*width))
	{
		//Covers West Edge
		gx = 0.f;
		gy = 0.f;
		srcPosition=y-(*width); //This makes it go one row below what it's currently at
		for (mPosition = 0; mPosition < 9; srcPosition+=(*width)) //This makes use of the matrix position as a check to see if everything is done
		{
			gx += src[srcPosition] * (mx[mPosition]+mx[mPosition+1]); //This factoring reduces multiplication, an expensive operation
			gy += src[srcPosition++] * (my[mPosition]+my[mPosition+1]);
			mPosition++; //This is what increments the position in the array it needs to multiply by
			gx+=src[srcPosition]*mx[++mPosition];
			gy+=src[srcPosition--]*my[mPosition++];
		}
		srcPosition=y; //This resets the srcPosition back to what it should be
		g_mag[srcPosition] = hypotf(gy, gx);
		g_ang[srcPosition++] = atan2f(gy, gx); //The ++ makes srcPosition accurate for the next iterations
		//End of Covering West Edge
		for (x = 1; x <maxColumnLimit; x++)
		{
			gx = 0.f;
			gy = 0.f;
			srcPosition-=(*width); //This makes the row used be below the current row
			for (mPosition=0; mPosition< 9; srcPosition+=(*width))
			{
				gx += src[--srcPosition] * mx[mPosition]; //This uses the same principles as before but no factoring can be done
				gy += src[srcPosition++] * my[mPosition++];
				gx += src[srcPosition] * mx[mPosition];
				gy += src[srcPosition++] * my[mPosition++];
				gx += src[srcPosition] * mx[mPosition];
				gy += src[srcPosition--] * my[mPosition++];//All the -- and ++ operators do is essentially reset the x coordinate every iteration
			}
			srcPosition-=doubleWidth; //This reset makes it so srcPosition can also manage to always have the correct x coordinate
			g_mag[srcPosition] = hypotf(gy, gx);
			g_ang[srcPosition++] = atan2f(gy, gx);//The srcPosition then increments along with x 
		}
		//Covers East Edge
		gx = 0.f;
		gy = 0.f;
		srcPosition-=(*width);//The x coordinate is already correct from the previous iteration.
		for (mPosition=0; mPosition < 9; srcPosition+=(*width))
		{
			gx+=src[--srcPosition]*mx[mPosition];
			gy+=src[srcPosition++]*my[mPosition++];
			gx+=src[srcPosition]*(mx[mPosition]+mx[mPosition+1]);
			gy+=src[srcPosition]*(my[mPosition]+my[mPosition+1]);
			mPosition+=2;
		}
		srcPosition-=doubleWidth;//This resets it to what it should be 
		g_mag[srcPosition] = hypotf(gy, gx);
		g_ang[srcPosition] = atan2f(gy, gx);
		//Finishes covering east edge
	}
}

int is_edge(float* g_mag, float* g_ang, float* threshold, const int* x, const int* y, const int* width, const int* maxColumnLimit,const int* maxRowLimit)
{
	int position=(*y)+(*x); //This makes a much better reference to the current position in the array
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
		int belowLimit=(((*y)<(*maxRowLimit))?(*width)+position:position); //These two expressions simplify a lot of calculations
		int aboveLimit=(((*y)>0  )?position-(*width):position); //by determing edge conditions or not
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
	const int size=height*sizeof(float); //This gives the array size
	const int maxRowLimit=height-width; //This reduces the number of computations by establishing a limit
	const int maxColumnLimit=width-1; //This reduces computation
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

// new record     detect_edges: 0.028357 seconds







