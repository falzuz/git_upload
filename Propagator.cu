__global__ void calculate_drift(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, cuDoubleComplex *kin, cuDoubleComplex *kin_conjug, cuDoubleComplex *dippot,cuDoubleComplex *drift, cuDoubleComplex *drift_conjug, double *random)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);

	cuDoubleComplex tempm=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp=make_cuDoubleComplex(0.,0.);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		tempm=tempm+psi_conjug[ind(i,j,k,l,m)]  *psi[ind(i,j,k,l,m-1)];
		tempp=tempp+psi_conjug[ind(i,j,k,l,m+1)]*psi[ind(i,j,k,l,m)];
	}
	
	double V=OMEGAX*(double(j)-double(XSIZE)/2.+0.5)*(double(j)-double(XSIZE)/2.+0.5)+OMEGAY*(double(k)-double(YSIZE)/2.+0.5)*(double(k)-double(YSIZE)/2.+0.5)+OMEGAZ*(double(l)-double(ZSIZE)/2.+0.5)*(double(l)-double(ZSIZE)/2.+0.5);

	for(int i=0; i<COMPONENTS; i++)
	{		
		drift[ind(i,j,k,l,m)]=DT*(					
										//Berry phase term
												  psi[ind(i,j,k,l,m-1)]-psi[ind(i,j,k,l,m)]
										  +EPS*(
										//chemical potential term 
												+ (MU-V)*psi[ind(i,j,k,l,m-1)]
										//kinetic term
												+kin[ind(i,j,k,l,m-1)]
										//interaction term
												- G*tempm*psi[ind(i,j,k,l,m-1)]
										//dipole term
												-dippot[ind(i,j,k,l,m-1)]*psi[ind(i,j,k,l,m-1)]
												)
									 )
							     //noise
								 +make_cuDoubleComplex(random[ind_rand(i,j,k,l,m,0)],random[ind_rand(i,j,k,l,m,1)]);
								 
		drift_conjug[ind(i,j,k,l,m)]=DT*(					
										//Berry phase term
												  psi_conjug[ind(i,j,k,l,m+1)]-psi_conjug[ind(i,j,k,l,m)]
										  +EPS*(
										//chemical potential term 
												+ (MU-V)*psi_conjug[ind(i,j,k,l,m+1)]
										//kinetic term
												+kin_conjug[ind(i,j,k,l,m+1)]
										//interaction term
												- G*tempp*psi_conjug[ind(i,j,k,l,m+1)]
										//dipole term
												-dippot[ind(i,j,k,l,m)]*psi_conjug[ind(i,j,k,l,m+1)]
												)
									 )
							     //noise
								 +make_cuDoubleComplex(random[ind_rand(i,j,k,l,m,0)],-random[ind_rand(i,j,k,l,m,1)]);
	}
}



__global__ void calculate_drift_without_noise(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, cuDoubleComplex *kin, cuDoubleComplex *kin_conjug, cuDoubleComplex *dippot,cuDoubleComplex *drift, cuDoubleComplex *drift_conjug, double factor)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);

	cuDoubleComplex tempm=make_cuDoubleComplex(0.,0.);
	cuDoubleComplex tempp=make_cuDoubleComplex(0.,0.);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		tempm=tempm+psi_conjug[ind(i,j,k,l,m)]  *psi[ind(i,j,k,l,m-1)];
		tempp=tempp+psi_conjug[ind(i,j,k,l,m+1)]*psi[ind(i,j,k,l,m)];
	}
	
	double V=OMEGAX*(double(j)-double(XSIZE)/2.+0.5)*(double(j)-double(XSIZE)/2.+0.5)+OMEGAY*(double(k)-double(YSIZE)/2.+0.5)*(double(k)-double(YSIZE)/2.+0.5)+OMEGAZ*(double(l)-double(ZSIZE)/2.+0.5)*(double(l)-double(ZSIZE)/2.+0.5);


	for(int i=0; i<COMPONENTS; i++)
	{		
		drift[ind(i,j,k,l,m)]=factor*DT*(					
										//Berry phase term
												  psi[ind(i,j,k,l,m-1)]-psi[ind(i,j,k,l,m)]
										  +EPS*(
										//chemical potential term 
												+ (MU-V)*psi[ind(i,j,k,l,m-1)]
										//kinetic term
												+kin[ind(i,j,k,l,m-1)]
										//interaction term
												- G*tempm*psi[ind(i,j,k,l,m-1)]
										//dipole term
												-dippot[ind(i,j,k,l,m-1)]*psi[ind(i,j,k,l,m-1)]
												)
									 );
								 
		drift_conjug[ind(i,j,k,l,m)]=factor*DT*(					
										//Berry phase term
												  psi_conjug[ind(i,j,k,l,m+1)]-psi_conjug[ind(i,j,k,l,m)]
										  +EPS*(
										//chemical potential term 
												+ (MU-V)*psi_conjug[ind(i,j,k,l,m+1)]
										//kinetic term
												+kin_conjug[ind(i,j,k,l,m+1)]
										//interaction term
												- G*tempp*psi_conjug[ind(i,j,k,l,m+1)]
										//dipole term
								   -dippot[ind(i,j,k,l,m)]*psi_conjug[ind(i,j,k,l,m+1)]												
												)
									 );
	}
}


__global__ void compute_kinetic_part(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug,cuDoubleComplex *result, cuDoubleComplex *result_conjug)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	double px,py,pz;
	
	if(j<XSIZE/2||XSIZE==1)
	{
		px=2.*M_PI*double(j)/double(XSIZE);
	}
	else
	{
		px=-2.*M_PI*double(XSIZE-j)/double(XSIZE);
	}
	
	if(k<YSIZE/2||YSIZE==1)
	{
		py=2.*M_PI*double(k)/double(YSIZE);
	}
	else
	{
		py=-2.*M_PI*double(YSIZE-k)/double(YSIZE);
	}
			
	if(l<ZSIZE/2||ZSIZE==1)
	{
		pz=2.*M_PI*double(l)/double(ZSIZE);
	}
	else
	{
		pz=-2.*M_PI*double(ZSIZE-l)/double(ZSIZE);		
	}
	
	double p2=px*px+py*py+pz*pz;

	/*double p2=4.*(mysquare(sin(M_PI*double(j)/double(XSIZE)))+mysquare(sin(M_PI*double(k)/double(YSIZE)))+mysquare(sin(M_PI*double(l)/double(ZSIZE))));
	double pz2=4.*mysquare(sin(M_PI*double(l)/double(ZSIZE)));*/
	
	for(int i=0; i<COMPONENTS; i++)
	{		
		result[ind(i,j,k,l,m)]=-p2*psi[ind(i,j,k,l,m)];
		result_conjug[ind(i,j,k,l,m)]=-p2*psi_conjug[ind(i,j,k,l,m)];
	}
}

__global__ void create_density(cuDoubleComplex *psi, cuDoubleComplex *psi_conjug, cuDoubleComplex *density)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	for(int i=0; i<COMPONENTS; i++)
	{
		density[ind(i,j,k,l,m)]=psi_conjug[ind(i,j,k,l,m+1)]*psi[ind(i,j,k,l,m)];
	}
}


__global__ void create_dipgrid(cuDoubleComplex *dipgrid)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=blockDim.z * blockIdx.z + threadIdx.z;
	
	double rx,ry,rz;
	
	if(j<int(XSIZE/2))
	{
		rx=double(j);
	}
	else
	{
		rx=double(j-XSIZE);
	}
	if(k<int(YSIZE/2))
	{
		ry=double(k);
	}
	else
	{
		ry=double(k-YSIZE);
	}
	if(l<int(ZSIZE/2))
	{
		rz=double(l);
	}
	else
	{
		rz=double(l-ZSIZE);
	}
	
	
	double result;
	
	double r2=rx*rx+ry*ry+rz*rz;
	
	if(j==0&&k==0&&l==0)
	{
		result=0.;
	}
	else
	{
		result=M/4./M_PI*(1.-3. * ( rx*rx * SIN2 + rz*rz * COS2 + rx*rz * SINCOS) / r2)/pow(r2,1.5);
	}
		
	for(int i=0; i<COMPONENTS; i++)
	{
		dipgrid[ind(i,j,k,l,0)]=make_cuDoubleComplex(result,0.);
	}
	
}

__global__ void multiply_by_dipole_potential(cuDoubleComplex *data, cuDoubleComplex *dipgrid)
{
	const int j=blockDim.x * blockIdx.x + threadIdx.x;
	const int k=blockDim.y * blockIdx.y + threadIdx.y;
	const int l=int((blockDim.z * blockIdx.z + threadIdx.z)/TAUSIZE);
	const int m=int((blockDim.z * blockIdx.z + threadIdx.z)%TAUSIZE);
	
	
	for(int i=0; i<COMPONENTS; i++)
	{
		data[ind(i,j,k,l,m)]=dipgrid[ind(i,j,k,l,0)]*data[ind(i,j,k,l,m)];
	}
}

class Propagator{
	public:
		Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug);
		Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug, unsigned long long seed);
		~Propagator();
		void propagate();
		void propagate_without_noise(double factor);
	private:
		ComplexLattice *fields, *fields_conjug;
		ComplexLattice *drift, *drift_conjug;
		ComplexLattice *kin, *kin_conjug;
		ComplexLattice *dippot;
		ComplexLatticeSpatial *dipgrid;
		dim3 dimblock;
		dim3 dimgrid;
		dim3 dimblock2;
		dim3 dimgrid2;
		curandGenerator_t gen;
		double *randomstore;	
};

	
Propagator::Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug)
{
	fields=lattice_fields;
	fields_conjug=lattice_fields_conjug;
	drift=lattice_drift;
	drift_conjug=lattice_drift_conjug;
	
	if((XSIZE%BLOCKSIZEX !=0)||
	   (YSIZE%BLOCKSIZEY !=0)||
	   (ZSIZE%BLOCKSIZEZ !=0)||
	   (TAUSIZE%BLOCKSIZETAU !=0))
	{
		cout<<"Length of the lattice dimensions must be a multiple of respective block size!"<<endl;
	}
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;
	
	dim3 temp_dimblock2(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ);
	dim3 temp_dimgrid2(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE/BLOCKSIZEZ));
	dimblock2=temp_dimblock2;
	dimgrid2=temp_dimgrid2;
	
	kin=new ComplexLattice(false);
	kin_conjug=new ComplexLattice(true);
	
	dippot=new ComplexLattice(false);
	
	dipgrid=new ComplexLatticeSpatial(false);
	
	create_dipgrid<<<dimgrid2,dimblock2>>>(dipgrid->get_pointer());
	
	dipgrid->fft();
	
	cudaMalloc(&randomstore, 2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
//	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
}

Propagator::Propagator(ComplexLattice *lattice_fields,ComplexLattice *lattice_fields_conjug, ComplexLattice *lattice_drift, ComplexLattice *lattice_drift_conjug, unsigned long long seed)
{
	fields=lattice_fields;
	fields_conjug=lattice_fields_conjug;
	drift=lattice_drift;
	drift_conjug=lattice_drift_conjug;
	
	if((XSIZE%BLOCKSIZEX !=0)||
	   (YSIZE%BLOCKSIZEY !=0)||
	   (ZSIZE%BLOCKSIZEZ !=0)||
	   (TAUSIZE%BLOCKSIZETAU !=0))
	{
		cout<<"Length of the lattice dimensions must be a multiple of respective block size!"<<endl;
	}
	dim3 temp_dimblock(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ*BLOCKSIZETAU);
	dim3 temp_dimgrid(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE*TAUSIZE/(BLOCKSIZEZ*BLOCKSIZETAU)));
	dimblock=temp_dimblock;
	dimgrid=temp_dimgrid;
	
	dim3 temp_dimblock2(BLOCKSIZEX ,BLOCKSIZEY, BLOCKSIZEZ);
	dim3 temp_dimgrid2(int(XSIZE/BLOCKSIZEX), int(YSIZE/BLOCKSIZEY), int(ZSIZE/BLOCKSIZEZ));
	dimblock2=temp_dimblock2;
	dimgrid2=temp_dimgrid2;
	
	kin=new ComplexLattice(false);
	kin_conjug=new ComplexLattice(true);

	dippot=new ComplexLattice(false);
	
	dipgrid=new ComplexLatticeSpatial(false);
	
	create_dipgrid<<<dimgrid2,dimblock2>>>(dipgrid->get_pointer());
	
	dipgrid->fft();
	
	cudaMalloc(&randomstore, 2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE*sizeof(double));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
	curandSetPseudoRandomGeneratorSeed(gen, seed);
}

Propagator::~Propagator()
{
	cudaFree(randomstore);
	curandDestroyGenerator(gen);
	delete kin;
	delete kin_conjug;
	delete dippot;
	delete dipgrid;
}

void Propagator::propagate()
{
	*kin=*fields;
	*kin_conjug=*fields_conjug;

	kin->fft();
	kin_conjug->fft();
	
	cudaDeviceSynchronize();
	compute_kinetic_part<<<dimgrid,dimblock>>>(kin->get_pointer(),kin_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer());
	cudaDeviceSynchronize();
	
	kin->fft_inv();
	kin_conjug->fft_inv();
	
	kin->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	kin_conjug->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	create_density<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),dippot->get_pointer());
	
	cudaDeviceSynchronize();
	dippot->fft();
	
	cudaDeviceSynchronize();
	multiply_by_dipole_potential<<<dimgrid,dimblock>>>(dippot->get_pointer(),dipgrid->get_pointer());
	
	cudaDeviceSynchronize();
	dippot->fft_inv();
	
	cudaDeviceSynchronize();
	dippot->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	curandGenerateNormalDouble(gen,randomstore,2*COMPONENTS*XSIZE*YSIZE*ZSIZE*TAUSIZE,0.,sqrt(DT));
	cudaDeviceSynchronize();
	
	calculate_drift<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer(),dippot->get_pointer(),drift->get_pointer(),drift_conjug->get_pointer(), randomstore);
	cudaDeviceSynchronize();
	
	cudaDeviceSynchronize();
	*fields+=*drift;
	cudaDeviceSynchronize();
	*fields_conjug+=*drift_conjug;
	cudaDeviceSynchronize();
}

void Propagator::propagate_without_noise(double factor)
{
	*kin=*fields;
	*kin_conjug=*fields_conjug;

	kin->fft();
	kin_conjug->fft();
	
	cudaDeviceSynchronize();
	compute_kinetic_part<<<dimgrid,dimblock>>>(kin->get_pointer(),kin_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer());
	cudaDeviceSynchronize();
	
	kin->fft_inv();
	kin_conjug->fft_inv();
	
	kin->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	kin_conjug->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	create_density<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),dippot->get_pointer());
	
	cudaDeviceSynchronize();
	dippot->fft();
	
	cudaDeviceSynchronize();
	multiply_by_dipole_potential<<<dimgrid,dimblock>>>(dippot->get_pointer(),dipgrid->get_pointer());
	
	cudaDeviceSynchronize();
	dippot->fft_inv();
	
	cudaDeviceSynchronize();
	dippot->normalize(1./double(XSIZE*YSIZE*ZSIZE));
	
	cudaDeviceSynchronize();
	calculate_drift_without_noise<<<dimgrid,dimblock>>>(fields->get_pointer(),fields_conjug->get_pointer(),kin->get_pointer(),kin_conjug->get_pointer(),dippot->get_pointer(),drift->get_pointer(),drift_conjug->get_pointer(),factor);
	cudaDeviceSynchronize();
	
	cudaDeviceSynchronize();
	*fields+=*drift;
	cudaDeviceSynchronize();
	*fields_conjug+=*drift_conjug;
	cudaDeviceSynchronize();
}
