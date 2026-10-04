#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <curand.h>
#include <cuComplex.h>
#include <cufft.h>
#include <iostream>
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <complex>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <sstream>
#include <fstream>
#include <errno.h>
#include <filesystem>

// only to have getopt
#include <getopt.h>

using namespace std;

#include "Parameters.h"
#include "Helpers.cu"
#include "Lattice.cu"
#include "Propagator.cu"
#include "Observables.cu"
#include "SaveResults.cu"

#define mf 0

#ifndef SIMULATION_SEED
#define SIMULATION_SEED 1948175546068024ULL
#endif
constexpr unsigned long long seedvalue = SIMULATION_SEED;


int main(int argc, char** argv)
{
		
	//////////////
	// SETTINGS //
	//////////////
	
	int devnum = 0;   /////// !!!!!!!!!!! HAS TO BE REMOVED !	
	cudaError_t device_status = cudaSetDevice(devnum);
	if (device_status != cudaSuccess) {
		cerr << "CUDA initialization failed: " << cudaGetErrorString(device_status) << endl;
		return 1;
	}
	double timer=time(NULL);
	

	struct which_observables which;
	which.n_tot		=	true	;
	which.P2		=	false	;	
	which.jrot2		=	false	;
	which.spectrum	=	false	;
	which.dispersion=	false	;
	which.dd		=	false	;
	which.dd_y		=	false	;
	which.dd_x		=	false	;
	which.anomalous_spectrum	=	false	;
	which.density_tot=	false	;
	which.spectrum_tot=	false	;
	which.drift		=	false	;
	which.max_momentum= M_PI * sqrt(3.0);
	which.bins		=	300		;
	which.min_drift	=	0.		;	
	which.max_drift	=	100		;	
	which.drift_bins=	2000	;

	
	ComplexLattice *fields=new ComplexLattice(false);
	ComplexLattice *fields_conjug=new ComplexLattice(true);
	
	ComplexLattice *drift=new ComplexLattice(false);
	ComplexLattice *drift_conjug=new ComplexLattice(true);

	ComplexLatticeHost *fields_host=new ComplexLatticeHost();
	ComplexLatticeHost *fields_conjug_host=new ComplexLatticeHost();

	Propagator *prop=new Propagator(fields,fields_conjug,drift,drift_conjug,seedvalue);
	Observables *obs=new Observables(fields,fields_conjug,which);
	
	SaveResults * save= new SaveResults(obs,seedvalue);
	if(!save->make_directory()){
		return 1;
	}
	save->save_parameters();
	if (!save->make_directory_rawdata()) {
		return 1;
	}


	
	if(mf==0 && G != 0)
	{
	    fields->set_mean_field(sqrt(MU/G/COMPONENTS),0.);
	    fields_conjug->set_mean_field(sqrt(MU/G/COMPONENTS),0.);
	}
	if(mf==1)
	{
	    double init = 0.;
		double imag_init = 0.;
		fields->set_mean_field(init, imag_init);
	    fields_conjug->set_mean_field(init, -imag_init);
	}
	if(mf==2){

		save->read_lattice(fields_host, fields_conjug_host, 
							//"./DATA/BoseCL_Simulation_T_0.416667_mu_0.2_g_0.729834_M_0_SIGMA_0.6_D_14.5196_OMEGAX_4e-05_OMEGAY_4e-05_OMEGAZ_0_components_1_lattice_384_384_1_48/rawdata/Lattice_80.txt");
							//"./DATA/BoseCL_Simulation_T_10_mu_3_g_0.1_M_0.6_SIGMA_0.6_OMEGAX_0.01358_OMEGAY_0.00123_OMEGAZ_0.34394_components_1_lattice_96_372_28_8/Run1/rawdata/Lattice_2000.txt");
							//"./Lattice_start.txt"); // mu 3 Run2
							"/mnt/sds-hd/sd19k005/luca/PAPER/DATA/BoseCL_Simulation_T_4.46429_mu_3_g_0.42282_M_1.4948_SIGMA_0.87266_OMEGAX_0.00306_OMEGAY_0.001_OMEGAZ_0.03752_components_1_lattice_200_220_84_28/Run20/rawdata/Lattice_3000.txt"

		);

		fields_host->copy_to_device(fields);
		fields_conjug_host->copy_to_device(fields_conjug);

		cout << "Lattice loaded from file" << endl;
	
	}

	
	cout<<"Started simulation with parameters"<<endl
		  <<"T "<<1./(TAUSIZE*EPS)<<endl
		  <<"mu "<<MU<<endl
		  <<"g "<<G<<endl
		  <<"M "<<M<<endl
		  <<"OMEGAX "<<OMEGAX<<endl
		  <<"OMEGAY "<<OMEGAY<<endl
		  <<"OMEGAZ "<<OMEGAZ<<endl
		  <<"components "<<COMPONENTS<<endl
		  <<"lattice "<<XSIZE<<"x"<<YSIZE<<"x"<<ZSIZE<<"x"<<TAUSIZE<<endl
		  <<"dt "<<DT<<endl
		  <<"steps "<<STEPS<<endl
		  <<"seed "<<seedvalue<<endl
		  <<"mf "<<mf<<endl;

	
	
	for(int i=0; i<STEPS; i++)
	{
		if(i%SNAPSHOT_TRIGGER==0||i==1)
		{
			save->save_snapshot();
		}
		
		
		if(i%RAW_TRIGGER==0 && i >= 1) //|| i>= STEPS-SNAPSHOT_TRIGGER)
		{
		    fields_host->copy_from_device(fields);
		    fields_conjug_host->copy_from_device(fields_conjug);
		    save->save_lattice(fields_host,fields_conjug_host);
		} 
        
		
		prop->propagate();
		//prop->propagate_without_noise(1.);

		
		if(i%EVALUATE_TRIGGER==0){
			obs->evaluate();
		}
		
  
	}
	
	
	save->save_snapshot();

	cout<<"Finished simulation after computation time of "<<time(NULL)-timer<<" seconds."<<endl;

	return 0;
}
