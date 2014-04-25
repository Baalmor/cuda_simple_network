
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <windows.h>
#include <cstdlib>
#include <stdio.h>
#include <conio.h>
#include <time.h>
#include <math.h>

using namespace std;

#define CUDA_CHECK(x) do { if((x) != cudaSuccess) {                                                         \
  printf("Error: %s at line %d. error code: \"%s\"\n",__FILE__,__LINE__,cudaGetErrorString(x));       \
  exit(x);}} while(0)


/*
 this program is doing extremely simple neural net with GROUP_SIZE/SUB_GROUP_SIZE layers.
every neuron binded with whole previous layer but every connection have diffirent weight. 
There is a "layer calculating" parallelism tactic. Every whole layer calculated at once.
Bitmaps map also calculated in multithread.
*/



void SaveBitmapToFile( BYTE* pBitmapBits,  
                                       LONG lWidth,  
                                       LONG lHeight,  
                                       WORD wBitsPerPixel,  
                                       LPCTSTR lpszFileName )  
{  
    unsigned long headers_size = sizeof( BITMAPFILEHEADER ) +  
                                 sizeof( BITMAPINFOHEADER );  
    unsigned long padding_size    = ( 4 - ( ( lWidth * 3 ) % 4 ) ) % 4;  
    unsigned long pixel_data_size = lHeight * ( ( lWidth * 3 ) + padding_size );  
    BITMAPINFOHEADER bmpInfoHeader = {0};   
    bmpInfoHeader.biSize = sizeof(BITMAPINFOHEADER);  
    bmpInfoHeader.biBitCount = wBitsPerPixel;  
    bmpInfoHeader.biClrImportant = 0;  
    bmpInfoHeader.biClrUsed = 0;  
    bmpInfoHeader.biCompression = BI_RGB;  
    bmpInfoHeader.biHeight = lHeight;  
    bmpInfoHeader.biWidth = lWidth;  
    bmpInfoHeader.biPlanes = 1;  
    bmpInfoHeader.biSizeImage = pixel_data_size;  

    BITMAPFILEHEADER bfh = {0};  
    bfh.bfType=0x4D42;  
    bfh.bfOffBits = headers_size;  
    bfh.bfSize =  headers_size + pixel_data_size;  
    HANDLE hFile = CreateFile( lpszFileName,  
                               GENERIC_WRITE,  
                               0,  
                               NULL,  
                               CREATE_ALWAYS,  
                               FILE_ATTRIBUTE_NORMAL,  
                               NULL );  
 
    if( !hFile ) return;  
  
    DWORD dwWritten = 0;  

    WriteFile( hFile,  
               &bfh,  
               sizeof(bfh),  
               &dwWritten ,  
               NULL );  
 
    WriteFile( hFile,  
               &bmpInfoHeader,  
               sizeof(bmpInfoHeader),  
               &dwWritten,  
               NULL );  

    WriteFile( hFile,  
               pBitmapBits,  
               bmpInfoHeader.biSizeImage,  
               &dwWritten,  
               NULL );  
 
    CloseHandle( hFile );  
}  

/////start program

/*///////
main idea to decompose structure in 1-dimensional array
so the group is 
i - number of neuron
j - number of group
SUB_GROUP_SIZE*j>i>SUB_GROUP_SIZE*(j+1)
[i+0]activation
[i+1]potential
[i+2]threshold
[i+3]...[i+SUB_GROUP_SIZE] - weights
///////*/

#define GROUP_SIZE 2048 //count of neurons in net
#define SUB_GROUP_SIZE 256 //size of layer (must be divisible with GROUP_SIZE)
#define SUB_GROUP_OFFSET 3 //activation //potential //threshold

//prepere memory
float host_group[GROUP_SIZE*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)];
const int FULL_STRUCT_SIZE=GROUP_SIZE*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET);


//for timing
class timer {
	private:
		unsigned long begTime;
	public:
		void reset() {
			begTime = clock();
		}

		unsigned long elapsedTime() {
			return ((unsigned long) clock() - begTime);
		}


		bool isTimeout(unsigned long seconds) {
			return seconds >= elapsedTime();
		}
};



void setWeightsAndThresholds(float* group)
{
	for(int j=0;j<GROUP_SIZE;j++)
    {
		group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+2]=rand()%1000/1000.0f;
		for(int i=0;i<SUB_GROUP_SIZE;i++)
		{
		group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+i+SUB_GROUP_OFFSET]=rand()%1000/1000.0f;
		}
    }
}


void printGroup(float* group)
{
	for(int j=0;j<GROUP_SIZE;j++)
    {
		float *groupPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)];
		printf(" a:%0.1f |",*groupPtr);
		printf(" p:%0.3f |",*(groupPtr+1));
		printf(" t:%0.3f |",*(groupPtr+2));
		for(int i=0;i<SUB_GROUP_SIZE;i++)
		{
		printf(" %0.3f |",*(groupPtr+i+SUB_GROUP_OFFSET));
		}
		printf("\n");
    }
}

__global__ void setInputs(float* group,float* f)
{
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	if(j>=SUB_GROUP_SIZE) 
		return;
	group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+1]=f[j];
	__syncthreads(); //must synchronize here because next call is kernel function and theoretically this function may not be finished before next function calling
}

__shared__ float prevResult[SUB_GROUP_SIZE];//array for saving previous group activations

__global__ void updateGroup(float* group,unsigned int layerId)
{
	int p = blockIdx.x*blockDim.x+threadIdx.x;
	if(p>=SUB_GROUP_SIZE) 
		return;
	int j=p+layerId*SUB_GROUP_SIZE; //set group offset
	float *firstGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)];
	float *PotentialPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+1];
	float *ThresholdGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+2];
	if(*firstGPtr>0)
	{
	*firstGPtr=0.0f;
	*PotentialPtr=0;
	}
	float dPotential=0;
	if(layerId>0)
	{
		int w_offset=0;
		for(int i=0;i<SUB_GROUP_SIZE;i++)
		{
			dPotential=dPotential +
				prevResult[i] * (*(firstGPtr+SUB_GROUP_OFFSET+w_offset)) / SUB_GROUP_SIZE; //must divede every input for keeping value between 0 and 1 
			w_offset++;
		}
		*PotentialPtr+=dPotential;
	}
	if(*PotentialPtr>*ThresholdGPtr)
	{
		*firstGPtr=1.0f;
	}
	prevResult[p]=*firstGPtr;
}


__global__ void calcBitmap(float* group, BYTE buf[],BYTE bufFires[])
{
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	if (j >= GROUP_SIZE)return;
	float *firstGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)];
	float *PotentialPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+1];
	float *ThresholdGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+2];
	int c =j*3;
	float f=*PotentialPtr / *ThresholdGPtr;
	if(f>1)f=1.0f;
	unsigned int val =  
              int(f*255 + 0.5);                           
  
        buf[ c + 0 ] = (BYTE) val;  
        buf[ c + 1 ] = (BYTE) val;  
        buf[ c + 2 ] = (BYTE) val; 
		//show fires only
		val =int(*firstGPtr*255 + 0.5);                           
  
        bufFires[ c + 0 ] = (BYTE) val;  
        bufFires[ c + 1 ] = (BYTE) val;  
        bufFires[ c + 2 ] = (BYTE) val;
}

void calcBitmapTest(float group[], BYTE buf[],BYTE bufFires[])
{
	for(int j=0;j<GROUP_SIZE;j++)
	{
		if (j >= GROUP_SIZE)return;
		float *firstGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)];
		float *PotentialPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+1];
		float *ThresholdGPtr=&group[j*(SUB_GROUP_SIZE+SUB_GROUP_OFFSET)+2];
		int c =j*3;
		float f=*PotentialPtr / *ThresholdGPtr;
		if(f>1)f=1.0f;
		unsigned int val =  
				  int(f*255 + 0.5);                           
  
			buf[ c + 0 ] = (BYTE) val;  
			buf[ c + 1 ] = (BYTE) val;  
			buf[ c + 2 ] = (BYTE) val; 
			//show only fires
			val =int(*firstGPtr*255 + 0.5);                           
  
			bufFires[ c + 0 ] = (BYTE) val;  
			bufFires[ c + 1 ] = (BYTE) val;  
			bufFires[ c + 2 ] = (BYTE) val;
	}
}

__host__ int main(int argc, char** argv) {
    
	//init timer
  int stime;
  long ltime;
  ltime = time(NULL);
  stime = (unsigned) ltime/2;
  srand(stime);

    
   printf("Program started...\n");

    //init group
    setWeightsAndThresholds(host_group);
	//printGroup(host_group);
    float* dev_group_ptr;

	CUDA_CHECK(cudaMalloc((void**)&dev_group_ptr, FULL_STRUCT_SIZE*sizeof(float)));
	CUDA_CHECK(cudaMemcpy(dev_group_ptr, host_group , FULL_STRUCT_SIZE*sizeof(float), cudaMemcpyHostToDevice));
	
	//for bitmap
	BYTE* buf = new BYTE[ GROUP_SIZE * 3];  
	BYTE* fbuf = new BYTE[ GROUP_SIZE * 3];  //test
	BYTE* pBuf;
	BYTE* pBufFires;
	CUDA_CHECK(cudaMalloc((void**)&pBuf, GROUP_SIZE*3*sizeof(BYTE)));
	CUDA_CHECK(cudaMalloc((void**)&pBufFires, GROUP_SIZE*3*sizeof(BYTE)));

	cudaEvent_t syncEvent;
	CUDA_CHECK(cudaEventCreate(&syncEvent));    //create event for sync
  
  
	float inputs[SUB_GROUP_SIZE];
	float* dev_ptr_float;

	CUDA_CHECK(cudaMalloc((void**)&dev_ptr_float, SUB_GROUP_SIZE*sizeof(float)));
	
	bool quit=false;
	timer t;
	t.reset();

	int gridSize = (int)ceil((float)GROUP_SIZE/SUB_GROUP_SIZE);

	 printf("You may look in neural_fires_map.bmp and neural_potentials_map.bmp for watching how this net is working.\n");
	 printf("Press any key to stop...\n");
	//main cicle
    while ( !quit )
    {
        if(t.elapsedTime() >= 1) //timer in milliseconds
		{
			for(unsigned int i=0;i<SUB_GROUP_SIZE;i++)
			{
				inputs[i]=rand()%1000/1000.0;
			}
			CUDA_CHECK(cudaMemcpy(dev_ptr_float, inputs , SUB_GROUP_SIZE*sizeof(float), cudaMemcpyHostToDevice));
			setInputs<<<1, SUB_GROUP_SIZE>>>(dev_group_ptr,dev_ptr_float); //insert new inputs
			for(unsigned int groupId=0;groupId<GROUP_SIZE/SUB_GROUP_SIZE;groupId++)
			{
				updateGroup<<<1, SUB_GROUP_SIZE,0>>>(dev_group_ptr,groupId);//for sub_group_size < 512 looks better to use one grid with many threads
				//this calls completely synchronous because they fire in the same stream so no need to call __syncthreads()
			}

			CUDA_CHECK(cudaMemcpy( host_group, dev_group_ptr, FULL_STRUCT_SIZE*sizeof(float), cudaMemcpyDeviceToHost )); 
			//printGroup(host_group);
			//print to console
			//system("cls");

			calcBitmap<<<gridSize, SUB_GROUP_SIZE>>>(dev_group_ptr,pBuf,pBufFires); 
			CUDA_CHECK(cudaEventRecord(syncEvent, 0));  
			CUDA_CHECK(cudaEventSynchronize(syncEvent));

			CUDA_CHECK(cudaMemcpy( buf, pBuf, GROUP_SIZE*3*sizeof(BYTE), cudaMemcpyDeviceToHost )); 
			 
			SaveBitmapToFile( (BYTE*) buf,  
                            SUB_GROUP_SIZE,  
                            GROUP_SIZE/SUB_GROUP_SIZE,  
                            24,  
                            "neural_potentials_map.bmp" ); 

			CUDA_CHECK(cudaMemcpy( buf, pBufFires, GROUP_SIZE*3*sizeof(BYTE), cudaMemcpyDeviceToHost )); 

			SaveBitmapToFile( (BYTE*) buf,  
                            SUB_GROUP_SIZE,  
                            GROUP_SIZE/SUB_GROUP_SIZE,  
                            24,  
                            "neural_fires_map.bmp" ); 

			t.reset();
		}
		else
		{
			    if(_kbhit() ) {
				  quit=true;
				}
		}
	}

	//clean memory

	cudaFree(dev_group_ptr);
	cudaFree(dev_ptr_float);
	cudaFree(pBuf);
	cudaFree(pBufFires);

	cudaEventDestroy(syncEvent);

	delete [] buf; 
	printf("Program successfuly stopped.\n");
    return 0;
}


