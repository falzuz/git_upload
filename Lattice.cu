//calculate index for complex lattice
__device__ inline int ind(int compindex,int xindex,int yindex, int zindex, int tauindex)
{
	return compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+map(tauindex,TAUSIZE)*XSIZE*YSIZE*ZSIZE+map(xindex,XSIZE)*YSIZE*ZSIZE+map(yindex,YSIZE)*ZSIZE+map(zindex,ZSIZE);
}

__device__ inline int ind_rand(int compindex,int xindex,int yindex, int zindex, int tauindex, int part)
{
	//return compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE*2+tauindex*XSIZE*YSIZE*ZSIZE*2+xindex*YSIZE*ZSIZE*2+yindex*ZSIZE+zindex*2+part;
	return compindex*2*XSIZE*YSIZE*ZSIZE*TAUSIZE+xindex*2*YSIZE*ZSIZE*TAUSIZE+yindex*2*ZSIZE*TAUSIZE+zindex*2*TAUSIZE+tauindex*2+part;
}

__device__ inline int indl(int compindex,int xindex,int yindex, int zindex, int tauindex)
{
	return compindex*(BLOCKSIZETAU+2)*(BLOCKSIZEX+2)*(BLOCKSIZEY+2)*(BLOCKSIZEZ+2)+(tauindex+1)*(BLOCKSIZEX+2)*(BLOCKSIZEY+2)*(BLOCKSIZEZ+2)+(xindex+1)*(BLOCKSIZEY+2)*(BLOCKSIZEZ+2)+(yindex+1)*(BLOCKSIZEZ+2)+zindex+1;
}


//the Laplacian is implemented in a "local" version, taking as coordinates block coordinates, and a "global" version, taking lattice coordinates

__device__ inline cuDoubleComplex laplacian(cuDoubleComplex *fields, int i, int j, int k, int l, int m)
{
	return fields[ind(i,j-1,k,l,m)]+fields[ind(i,j+1,k,l,m)]+fields[ind(i,j,k-1,l,m)]+fields[ind(i,j,k+1,l,m)]+fields[ind(i,j,k,l-1,m)]+fields[ind(i,j,k,l+1,m)]-6.*fields[ind(i,j,k,l,m)];
}

__device__ inline cuDoubleComplex laplacian_l(cuDoubleComplex *fields, int i, int j, int k, int l, int m)
{
	return fields[indl(i,j-1,k,l,m)]+fields[indl(i,j+1,k,l,m)]+fields[indl(i,j,k-1,l,m)]+fields[indl(i,j,k+1,l,m)]+fields[indl(i,j,k,l-1,m)]+fields[indl(i,j,k,l+1,m)]-6.*fields[indl(i,j,k,l,m)];
}


__global__ void summation(cuDoubleComplex *data1, cuDoubleComplex *data2)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	data1[x]=data1[x]+data2[x];
}

__global__ void multiplication(cuDoubleComplex *data1, cuDoubleComplex *data2)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	data1[x]=data1[x]*data2[x];
}

__global__ void normalization(cuDoubleComplex *data, double factor)
{
	const int x = blockDim.x * blockIdx.x + threadIdx.x;
	data[x]=factor*data[x];
}

__global__ void set_value(cuDoubleComplex *data, double re_psi, double im_psi, bool conjug)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	cuDoubleComplex value;
	if (conjug)
	{
		value=make_cuDoubleComplex(re_psi,-im_psi);
	}
	else
	{
		value=make_cuDoubleComplex(re_psi,im_psi);
	}
	
	for(int i=0; i<COMPONENTS; i++)
	{
		data[ind(i,j,k,l,m)]=value;
	}
}

class ComplexLattice
{
	public:
		ComplexLattice(bool conjug);
		~ComplexLattice();
		void fft();
		void fft_inv();
		void normalize(double factor);
		void set_mean_field(double re_psi, double im_psi);
		void operator *=(ComplexLattice &other);
		void operator +=(ComplexLattice &other);
		void operator =(ComplexLattice &other);
		cuDoubleComplex *get_pointer();
	private:
		cuDoubleComplex *store;
		int length;
		bool conjugated;
		dim3 dimblock;
		dim3 dimgrid;
		dim3 dimblock2;
		dim3 dimgrid2;
		cufftHandle plan;
		void make_plan();
};

ComplexLattice::ComplexLattice(bool conjug)
{
	length=COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE;
	conjugated=conjug;
	cudaMalloc(&store,length*sizeof(cuDoubleComplex));
	
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY,BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;
	
	dim3 temp_dimblock2(BLOCKSIZE,1,1);
	dim3 temp_dimgrid2(int(length/BLOCKSIZE),1,1);
	dimblock2=temp_dimblock2;
	dimgrid2=temp_dimgrid2;
	
	make_plan();
}

ComplexLattice::~ComplexLattice()
{
	cudaFree(store);
	cufftDestroy(plan);
}



void ComplexLattice::make_plan()
{
	int rank = ZSIZE > 1 ? 3 : ((ZSIZE==1 && YSIZE > 1) ? 2 : ((ZSIZE==1 && YSIZE==1 && XSIZE>1) ? 1 : 0));
	int n[rank];

	if (rank == 1)
	{
		n[0] = XSIZE;
	}
	else if (rank == 2)
	{
		n[0] = XSIZE;
		n[1] = YSIZE;
	}
	else if (rank == 3)
	{
		n[0] = XSIZE;
		n[1] = YSIZE;
		n[2] = ZSIZE;
	}
	cufftPlanMany(&plan, rank, n, NULL, 0, 0, NULL, 0, 0, CUFFT_Z2Z, COMPONENTS*TAUSIZE);
}

void ComplexLattice::fft()
{
	if(conjugated)
	{
		cufftExecZ2Z(plan, store, store, CUFFT_FORWARD);
	}
	else
	{
		cufftExecZ2Z(plan, store, store, CUFFT_INVERSE);
	}
}

void ComplexLattice::fft_inv()
{
	if(conjugated)
	{
		cufftExecZ2Z(plan, store, store, CUFFT_INVERSE);
	}
	else
	{
		cufftExecZ2Z(plan, store, store, CUFFT_FORWARD);
	}
}

void ComplexLattice::normalize(double factor)
{
	cudaDeviceSynchronize();
	normalization<<<dimgrid2,dimblock2>>>(store, factor);
	cudaDeviceSynchronize();
}


void ComplexLattice::set_mean_field(double re_psi, double im_psi)
{
	cudaDeviceSynchronize();
	set_value<<<dimgrid,dimblock>>>(store,re_psi, im_psi, conjugated);
	cudaDeviceSynchronize();	
}

void ComplexLattice::operator +=(ComplexLattice &other)
{
	cudaDeviceSynchronize();
	summation<<<dimgrid2,dimblock2>>>(store, other.store);
	cudaDeviceSynchronize();
}

void ComplexLattice::operator *=(ComplexLattice &other)
{
	cudaDeviceSynchronize();
	multiplication<<<dimgrid2,dimblock2>>>(store, other.store);
	cudaDeviceSynchronize();
}

void ComplexLattice::operator =(ComplexLattice &other)
{
	cudaMemcpy(store,other.store,length*sizeof(cuDoubleComplex) ,cudaMemcpyDeviceToDevice);
}


cuDoubleComplex *ComplexLattice::get_pointer()
{
	return store;
}

























































class ComplexLatticeSpatial
{
	public:
		ComplexLatticeSpatial(bool conjug);
		~ComplexLatticeSpatial();
		void fft();
		void fft_inv();
		void normalize(double factor);
		void operator *=(ComplexLatticeSpatial &other);
		void operator +=(ComplexLatticeSpatial &other);
		void operator =(ComplexLatticeSpatial &other);
		cuDoubleComplex *get_pointer();
		int get_length();
	private:
		cuDoubleComplex *store;
		int length;
		bool conjugated;
		dim3 dimblock2;
		dim3 dimgrid2;
		cufftHandle plan;
		void make_plan();
};

ComplexLatticeSpatial::ComplexLatticeSpatial(bool conjug)
{
	length=COMPONENTS*XSIZE*YSIZE*ZSIZE;
	conjugated=conjug;
	cudaMalloc(&store,length*sizeof(cuDoubleComplex));
	
	
	dim3 temp_dimblock2(BLOCKSIZE,1,1);
	dim3 temp_dimgrid2(int(length/BLOCKSIZE),1,1);
	dimblock2=temp_dimblock2;
	dimgrid2=temp_dimgrid2;
	
	make_plan();
}

ComplexLatticeSpatial::~ComplexLatticeSpatial()
{
	cudaFree(store);
	cufftDestroy(plan);
}




void ComplexLatticeSpatial::make_plan()
{
	int rank = ZSIZE > 1 ? 3 : ((ZSIZE==1 && YSIZE > 1) ? 2 : ((ZSIZE==1 && YSIZE==1 && XSIZE>1) ? 1 : 0));
	int n[rank];

	if (rank == 1)
	{
		n[0] = XSIZE;
	}
	else if (rank == 2)
	{
		n[0] = XSIZE;
		n[1] = YSIZE;
	}
	else if (rank == 3)
	{
		n[0] = XSIZE;
		n[1] = YSIZE;
		n[2] = ZSIZE;
	}
	cufftPlanMany(&plan, rank, n, NULL, 0, 0, NULL, 0, 0, CUFFT_Z2Z, COMPONENTS);
}

void ComplexLatticeSpatial::fft()
{
	if(conjugated)
	{
		cufftExecZ2Z(plan, store, store, CUFFT_FORWARD);
	}
	else
	{
		cufftExecZ2Z(plan, store, store, CUFFT_INVERSE);
	}
}

void ComplexLatticeSpatial::fft_inv()
{
	if(conjugated)
	{
		cufftExecZ2Z(plan, store, store, CUFFT_INVERSE);
	}
	else
	{
		cufftExecZ2Z(plan, store, store, CUFFT_FORWARD);
	}
}

void ComplexLatticeSpatial::normalize(double factor)
{
	cudaDeviceSynchronize();
	normalization<<<dimgrid2,dimblock2>>>(store, factor);
	cudaDeviceSynchronize();
}


void ComplexLatticeSpatial::operator +=(ComplexLatticeSpatial &other)
{
	cudaDeviceSynchronize();
	summation<<<dimgrid2,dimblock2>>>(store, other.store);
	cudaDeviceSynchronize();
}

void ComplexLatticeSpatial::operator *=(ComplexLatticeSpatial &other)
{
	cudaDeviceSynchronize();
	multiplication<<<dimgrid2,dimblock2>>>(store, other.store);
	cudaDeviceSynchronize();
}

void ComplexLatticeSpatial::operator =(ComplexLatticeSpatial &other)
{
	cudaMemcpy(store,other.store,length*sizeof(cuDoubleComplex) ,cudaMemcpyDeviceToDevice);
}


cuDoubleComplex *ComplexLatticeSpatial::get_pointer()
{
	return store;
}

int ComplexLatticeSpatial::get_length()
{
	return length;
}





























class ComplexLatticeHost
{
	public:
		ComplexLatticeHost();
		~ComplexLatticeHost();
		void copy_from_device(ComplexLattice *device_lattice);
		void copy_to_device(ComplexLattice *device_lattice);
		complex <double> *get_pointer();
		int get_length();
	private:
		complex <double> *store;
		int length;	
};



ComplexLatticeHost::ComplexLatticeHost()
{
	length=COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE;

	store=new complex <double> [length];
}

ComplexLatticeHost::~ComplexLatticeHost()
{
	delete [] store;
}


void ComplexLatticeHost::copy_from_device(ComplexLattice *device_lattice)
{
	cudaMemcpy(store,device_lattice->get_pointer(),2*length*sizeof(double),cudaMemcpyDeviceToHost);
}

void ComplexLatticeHost::copy_to_device(ComplexLattice *device_lattice)
{
	cudaMemcpy(device_lattice->get_pointer(),store,2*length*sizeof(double),cudaMemcpyHostToDevice);
}

complex <double> *ComplexLatticeHost::get_pointer()
{
	return store;
}

int ComplexLatticeHost::get_length()
{
	return length;
}























class ComplexLatticeSpatialHost
{
	public:
		ComplexLatticeSpatialHost();
		~ComplexLatticeSpatialHost();
		void copy_from_device(ComplexLatticeSpatial *device_Lattice);
		void copy_to_device(ComplexLatticeSpatial *device_Lattice);
		complex <double> *get_pointer();
		int get_length();
	private:
		complex <double> *store;
		int length;	
};



ComplexLatticeSpatialHost::ComplexLatticeSpatialHost()
{
	length=COMPONENTS*XSIZE*YSIZE*ZSIZE;

	store=new complex <double> [length];
}

ComplexLatticeSpatialHost::~ComplexLatticeSpatialHost()
{
	delete [] store;
}


void ComplexLatticeSpatialHost::copy_from_device(ComplexLatticeSpatial *device_Lattice)
{
	cudaMemcpy(store,device_Lattice->get_pointer(),2*length*sizeof(double),cudaMemcpyDeviceToHost);
}

void ComplexLatticeSpatialHost::copy_to_device(ComplexLatticeSpatial *device_Lattice)
{
	cudaMemcpy(device_Lattice->get_pointer(),store,2*length*sizeof(double),cudaMemcpyHostToDevice);
}

complex <double> *ComplexLatticeSpatialHost::get_pointer()
{
	return store;
}

int ComplexLatticeSpatialHost::get_length()
{
	return length;
}