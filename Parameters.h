#ifndef PARAMETERS_H_
#define PARAMETERS_H_

//physical parameters

#define COMPONENTS 1

#define  XSIZE	 200
#define  YSIZE	 220
#define  ZSIZE	 84
#define  TAUSIZE	 28

#define  EPS	 0.008
#define  MU	 3.0
#define  G	 0.42282
#define  M	 1.4948
#define  SIGMA	 0.87266

#define  OMEGAX	 0.00306
#define  OMEGAY	 0.001
#define  OMEGAZ	 0.03752


//numerical parameters

#define STEPS  300100
#define DT     0.01
#define SNAPSHOT_TRIGGER   100000
#define EVALUATE_TRIGGER   1000
#define RAW_TRIGGER        100000




//computational parameters

#define BLOCKSIZE 256		//block size for calculations with one-dimensional grid (e. g. summation of lattices)
	
/*Block sizes for calculations with four-dimensional grid. Must be divisors of the lattice dimensions. If shared memory is used, choose as cubic as possible. 
*/ 
#define BLOCKSIZEX 4		
#define BLOCKSIZEY 4
#define BLOCKSIZEZ 4
#define BLOCKSIZETAU 4





#define SIN2 5.868195331118759572319731887546367943286895751953125000000000e-01
#define COS2 4.131804668881241537903292737610172480344772338867187500000000e-01
#define SINCOS 9.848093595620137641333258216036483645439147949218750000000000e-01



#endif



