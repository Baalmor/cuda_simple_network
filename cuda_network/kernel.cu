
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

#define GROUP_SIZE 512 //count of neurons in net
#define SUB_GROUP_SIZE 32 //size of layer (must be divisible with GROUP_SIZE)


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





class neuron
{
public:
    float in_weights[SUB_GROUP_SIZE];
    float potential;
    float threshold;
	float value;
    neuron();
	bool checked;
	void manualFire(float);
	__device__ void update();
	void mixWeights();
};

neuron::neuron()
{
    for (int i=0;i<SUB_GROUP_SIZE;i++)
    {
        in_weights[i]=0;
    }
    potential=0;
	value=0;
    //random threshold init
    threshold=rand()%1000/1000.0f;
	checked=false;
}






void neuron::mixWeights()
{
	int i=SUB_GROUP_SIZE;
	while(i--)
	{
		this->in_weights[i]=rand()%1000/1000.0f;
	}
}



void mixWeightsInGroup(neuron* group)
{
	for(int j=0;j<GROUP_SIZE;j++)
    {
		group[j].mixWeights();
    }
}

__global__ void setInputs(neuron* group,float f[])
{
	int j = threadIdx.x;
	if(j>=SUB_GROUP_SIZE) 
		return;
	group[j].potential=f[j];
}

__global__ void calcBitmap(neuron* group, BYTE buf[],BYTE bufFires[])
{
	int j = blockIdx.x*blockDim.x+threadIdx.x;
	if (j >= GROUP_SIZE)return;
	int c =j*3;
	float f=group[j].potential/group[j].threshold;
	if(f>1)f=1.0f;
	unsigned int val =  
              int(f*255 + 0.5);                           
  
        buf[ c + 0 ] = (BYTE) val;  
        buf[ c + 1 ] = (BYTE) val;  
        buf[ c + 2 ] = (BYTE) val; 
		//show only fires
		val =int(group[j].value*255 + 0.5);                           
  
        bufFires[ c + 0 ] = (BYTE) val;  
        bufFires[ c + 1 ] = (BYTE) val;  
        bufFires[ c + 2 ] = (BYTE) val;
}




__global__ void updateGroup(neuron* group,unsigned int layerId)
{
	int j = threadIdx.x+layerId*SUB_GROUP_SIZE;
	//group[j].update();
	if(group[j].value>0)
	{
	group[j].value=0.0f;
	group[j].potential=0;
	}
	float dPotential=0;
	if(layerId>0)
	{
		int offset=SUB_GROUP_SIZE*(layerId-1);
		int w_offset=0;
		for(int i=offset;i<(offset+SUB_GROUP_SIZE);i++)
		{
			dPotential=dPotential+group[i].value*group[j].in_weights[w_offset]/SUB_GROUP_SIZE; //must divede every input for keeping value between 0 and 1 
			w_offset++;
		}
		group[j].potential=group[j].potential+dPotential;
	}
	if(group[j].potential>group[j].threshold)
	{
		group[j].value=1.0f;
	}
}

void printNeuronsToBitmap(neuron* group,BYTE buf[])// for test
{
	int c=0;
	for(int j=0;j<GROUP_SIZE;j++)
    {
		unsigned char val =  
             0xBB;                         
  
        buf[ c + 0 ] = (BYTE) val;  
        buf[ c + 1 ] = (BYTE) val;  
        buf[ c + 2 ] = (BYTE) val;
		c+=3;
    }
}

//print to console
void printNeurons(neuron* group,int from=0,int to=0)
{
	for(int j=from;j<to;j++)
    {
		printf("%d (%f) %f %f %d = ",j,group[j].value,group[j].potential,group[j].threshold,group[j].checked);
		/*for(int k=0;k<SUB_GROUP_SIZE;k++)
		{
			printf("%d (%f): ",k,group[j].in_weights[k]);
		}*/
		printf("\n");
    }
}



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

neuron* createGroup()
{
	neuron* group;
    group = new neuron[GROUP_SIZE];
	return group;
}

__host__ int main(int argc, char** argv) {
    
	//init timer
  int stime;
  long ltime;
  ltime = time(NULL);
  stime = (unsigned) ltime/2;
  srand(stime);

    //init group
   neuron* host_group1;
   neuron* dev_group_ptr;

   host_group1 = createGroup();
   mixWeightsInGroup(host_group1);
   //printNeurons(host_group1,0,GROUP_SIZE);

   printf("Program started...\n");

	CUDA_CHECK(cudaMalloc((void**)&dev_group_ptr, GROUP_SIZE*sizeof(neuron)));
	CUDA_CHECK(cudaMemcpy(dev_group_ptr, host_group1 , GROUP_SIZE*sizeof(neuron), cudaMemcpyHostToDevice));
	
	//for bitmap
	BYTE* buf = new BYTE[ GROUP_SIZE * 3];  
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
				updateGroup<<<1, SUB_GROUP_SIZE>>>(dev_group_ptr,groupId);//for sub_group_size < 512 looks better to use one grid with many threads
				CUDA_CHECK(cudaEventRecord(syncEvent, 0));  
				CUDA_CHECK(cudaEventSynchronize(syncEvent));   //wait until threads finish updates layer
			}
			CUDA_CHECK(cudaMemcpy( host_group1, dev_group_ptr, GROUP_SIZE*sizeof(neuron), cudaMemcpyDeviceToHost )); 

			//print to console
			//system("cls");
			//printNeurons(host_group1,0,GROUP_SIZE);

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

	delete[] host_group1;
	delete [] buf; 
	printf("Program successfuly stopped.\n");
    return 0;
}


