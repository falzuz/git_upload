__device__ inline int map(int x, int m)
{
	if(x>=0&&x<m)
	{
		return x;
	}
	else if(x<0)
	{
		return x+m;
	}
	else
	{
		return x-m;
	}
}

__device__ inline double mysquare(double x)
{
	return x*x;
}

__device__ inline double mycube(double x)
{
	return x*x*x;
}

inline double mysquare_h(double x)
{
	return x*x;
}

inline double mycube_h(double x)
{
	return x*x*x;
}


void set_device_variable(double *devptr,double value)
{
	double temp=value;
	cudaMemcpy(devptr, &temp, sizeof(double), cudaMemcpyHostToDevice);
}

void set_multiple_device_variables(double *devptr,double value, int number)
{
	for(int i=0; i<number; i++)
	{
		set_device_variable(devptr+i,value);
	}
}	

double get_device_variable(double *devptr)
{
	double temp;
	cudaMemcpy(&temp, devptr, sizeof(double), cudaMemcpyDeviceToHost);
	return temp;
}


__device__ cuDoubleComplex  operator+(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a,b); }
__device__ cuDoubleComplex  operator-(cuDoubleComplex a, cuDoubleComplex b) { return cuCsub(a,b); }
__device__ cuDoubleComplex  operator*(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a,b); }
__device__ cuDoubleComplex  operator/(cuDoubleComplex a, cuDoubleComplex b) { return cuCdiv(a,b); }

__device__ cuDoubleComplex  operator*(double a, cuDoubleComplex b) { return make_cuDoubleComplex(a*b.x,a*b.y); }
