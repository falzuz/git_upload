//checks whether a folder exists
bool folder_exists(string foldername)
{
	struct stat st;
	if(stat(foldername.c_str(),&st) == 0){
		if(S_ISDIR(st.st_mode)== 0)
		{
			return false;
		}
		else
		{
			return true;
		}
    }
    else
    {
		return false;
	}
}


class SaveResults
{
	public:
		SaveResults(Observables *observables, unsigned long long seedval);
		bool make_directory();
		bool make_directory(string dirname);
		bool make_directory_rawdata();
		void save_parameters();
		void save_snapshot();
		void save_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug);
		void save_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int tauindex);
		void save_lattice_slices(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int min_tauindex,int max_tauindex);
		void read_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename);
		
		void read_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename);

		void save_lattice_Spatial(ComplexLatticeSpatialHost *lattice, ComplexLatticeSpatialHost *lattice_conjug);

		
	private:
		Observables *obs;
		struct which_observables which;
		string directory;
		string directory_rawdata;
		unsigned long long seedvalue;
};

SaveResults::SaveResults(Observables *observables, unsigned long long seedval)
{
	obs=observables;
	which=obs->get_which();
	seedvalue=seedval;
}


bool SaveResults::make_directory()
{
	stringstream dirname;

	dirname<<"./DATA/BoseCL_Simulation_T_"<<1./(TAUSIZE*EPS)
					<<"_mu_"<<MU<<"_g_"<<G<<"_M_"<<M<<"_SIGMA_"<<SIGMA<<"_OMEGAX_"<<OMEGAX<<"_OMEGAY_"<<OMEGAY<<"_OMEGAZ_"<<OMEGAZ<<"_components_"
					<<COMPONENTS<<"_lattice_"<<XSIZE<<"_"<<YSIZE<<"_"<<ZSIZE<<"_"<<TAUSIZE<<"/Run";
	
	return make_directory(dirname.str());
}

bool SaveResults::make_directory(string dirname)
{
    // Claim each run directory atomically; concurrent jobs retry the next one.
    try {
        const std::filesystem::path prefix(dirname);
        if (!prefix.parent_path().empty()) {
            std::filesystem::create_directories(prefix.parent_path());
        }
        for (int r = 1; r < 1000; ++r) {
            const std::filesystem::path candidate(dirname + to_string(r));
            if (std::filesystem::create_directory(candidate)) {
                directory = candidate.string();
                cout << "Directory created successfully: " << candidate << endl;
                return true;
            }
        }
        cerr << "No free run directory for " << dirname << endl;
    } catch (const std::filesystem::filesystem_error& e) {
        cerr << "Error: " << e.what() << endl;
    }
    return false;
}

bool SaveResults::make_directory_rawdata()
{
	directory_rawdata=directory+"/rawdata";
	if(mkdir(directory_rawdata.c_str(),0755)<0)
	{
		cout<<"Directory for raw data could not be created!"<<endl;
		return false;
	}
	else
	{
		cout<<"Directory for raw data was successfully created"<<endl;
		return true;
	}
}	

void SaveResults::save_parameters()
{
	fstream output;
	stringstream temp;
	temp<<directory<<"/Parameters.txt";		
	output.open(temp.str(),fstream::out);
	output<<"T\t"<<1./(TAUSIZE*EPS)<<endl
		  <<"mu\t"<<MU<<endl
		  <<"g\t"<<G<<endl
		  <<"M\t"<<M<<endl
		  <<"SIGMA\t"<<SIGMA<<endl
		  <<"OMEGAX\t"<<OMEGAX<<endl
		  <<"OMEGAY\t"<<OMEGAY<<endl
		  <<"OMEGAZ\t"<<OMEGAZ<<endl
		  <<"components\t"<<COMPONENTS<<endl
		  <<"lattice\t"<<XSIZE<<"x"<<YSIZE<<"x"<<ZSIZE<<"x"<<TAUSIZE<<endl
		  <<"dt\t"<<DT<<endl
		  <<"steps\t"<<STEPS<<endl
		  <<"SNAPSHOT_TRIGGER\t"<<SNAPSHOT_TRIGGER<<endl
		  <<"EVALUATE_TRIGGER\t"<<EVALUATE_TRIGGER<<endl
		  <<"seed\t"<<seedvalue;
	output.close();
	cout<<"Saved parameters"<<endl;
}

void SaveResults::save_snapshot()
{
	int number=obs->get_counter();
	//if (number == 0) return; // No measurements to average yet.
	
	fstream output;
	stringstream temp;

	/////////////
	// SCALARS //
	/////////////
	
	temp<<directory<<"/Scalars.txt";		
	output.open(temp.str(),fstream::app);
	
	double n_total, P2;
	string str_to_append;
	if(which.n_tot){
		obs->write_n_tot(&n_total);
		str_to_append = to_string(n_total) + "\t";
		
		if(number > 10 && isnan(n_total)) {
			cerr << "NaN value found in observable n_total." << endl;
			exit(1);			
		}
	}
	else{
		str_to_append = "\t\t";
	}

	if(which.P2){
		obs->write_P2(&P2);
		str_to_append = str_to_append + to_string(P2) + "\t";
	}

	else{
		str_to_append = str_to_append + "\t\t";
	}

	output << str_to_append << endl;


	//OLD SAVE
	/*
	fstream output_1;
	stringstream temp_1;

	temp_1<<directory<<"/Scalars_"<<number<<".txt";	
	output_1.open(temp_1.str(),fstream::out);
	
	//double n_total, P2;
	
	if(which.n_tot){
		obs->write_n_tot(&n_total);
		cout<<n_total;
		output_1<<"Total particle number: "<<n_total<<endl;
	}
	if(which.P2){
		obs->write_P2(&P2);
		cout<<P2;
  	output_1<<"Total momentum squared: "<<P2<<endl;
	}
	
	
	if(output.good() && output_1.good())
	*/
	if(output.good())
	{
		cout<<"Saved scalars at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving scalars at time "<<number<<" failed"<<endl;
	}
	output.close();
	// output_1.close();

	//////////////
	// SPECTRUM //
	//////////////

	if(which.spectrum)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum.txt";		
		output.open(temp.str(),fstream::app);
		double *spectrum=new double[which.bins];
		obs->write_spectrum(spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<spectrum[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		
		
		/* OLD
		temp<<directory<<"/Spectrum_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *spectrum=new double[which.bins];
		obs->write_spectrum(spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<spectrum[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}
	// SPECTRUM TOT (not angular averaging)
	if(which.spectrum_tot)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Spectrum_tot.txt";		
		output.open(temp.str(),fstream::app);
		double *spectrum_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_spectrum_tot(spectrum_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<spectrum_tot[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved angular-resolved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving angular-resolved spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();



		/*OLD
		temp<<directory<<"/Spectrum_tot_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *spectrum_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_spectrum_tot(spectrum_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<spectrum_tot[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved angular-resolved spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving angular-resolved spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	//  DENSITY TOT (not angular averaging)
	if(which.density_tot)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Density_tot.txt";		
		output.open(temp.str(),fstream::app);
		double *density_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_tot(density_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_tot[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved real-space density at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density at time "<<number<<" failed"<<endl;
		}
		output.close();

		/*OLD
		temp<<directory<<"/Spectrum_tot_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *density_tot=new double[XSIZE*YSIZE*ZSIZE];
		obs->write_density_tot(density_tot);
		for(int i=0; i<XSIZE*YSIZE*ZSIZE; i++)
		{
			output<<density_tot[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved real-space density at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving real-space density at time "<<number<<" failed"<<endl;
		}
		output.close();
		 */
	}

	// ANOMALOUS SPETRUM 
	if(which.anomalous_spectrum)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Anomalous_Spectrum.txt";		
		output.open(temp.str(),fstream::app);
		double *anomalous_spectrum=new double[which.bins];
		obs->write_anomalous_spectrum(anomalous_spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<anomalous_spectrum[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved anomalous spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving anomalous spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD
		temp<<directory<<"/Anomalous_Spectrum_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *anomalous_spectrum=new double[which.bins];
		obs->write_anomalous_spectrum(anomalous_spectrum);
		for(int i=0; i<which.bins; i++)
		{
			output<<anomalous_spectrum[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved anomalous spectrum at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving anomalous spectrum at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// DD 
	if(which.dd)
	{
		fstream output;
		stringstream temp;
		
		temp<<directory<<"/DensityDensity.txt";		
		output.open(temp.str(),fstream::app);
		double *dd=new double[which.bins];
		obs->write_dd(dd);
		for(int i=0; i<which.bins; i++)
		{
			output<<dd[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved dd at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dd at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/DensityDensity_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *dd=new double[which.bins];
		obs->write_dd(dd);
		for(int i=0; i<which.bins; i++)
		{
			output<<dd[i]<<endl;
		}
		
		if(output.good())
		{
			cout<<"Saved dd at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dd at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	
	// DD 
	if(which.dd_y)
	{
		fstream output;
		stringstream temp;
		
		temp<<directory<<"/DensityDensity_y.txt";		
		output.open(temp.str(),fstream::app);
		double *dd_y=new double[which.bins];
		obs->write_dd_y(dd_y);
		for(int i=0; i<which.bins; i++)
		{
			output<<dd_y[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved dd_y at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dd_y at time "<<number<<" failed"<<endl;
		}
		output.close();
	}

	// DD 
	if(which.dd_x)
	{
		fstream output;
		stringstream temp;
		
		temp<<directory<<"/DensityDensity_x.txt";		
		output.open(temp.str(),fstream::app);
		double *dd_x=new double[which.bins];
		obs->write_dd_x(dd_x);
		for(int i=0; i<which.bins; i++)
		{
			output<<dd_x[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved dd_x at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dd_x at time "<<number<<" failed"<<endl;
		}
		output.close();
	}
	
	// DISPERSION
	if(which.dispersion)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Dispersion.txt";		
		output.open(temp.str(),fstream::app);
		double *dispersion=new double[which.bins];
		obs->write_dispersion(dispersion);
		for(int i=0; i<which.bins; i++)
		{
			output<<dispersion[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved dispersion at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dispersion at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Dispersion_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *dispersion=new double[which.bins];
		obs->write_dispersion(dispersion);
		for(int i=0; i<which.bins; i++)
		{
			output<<dispersion[i]<<endl;
		}
		
		if(output.good())
		{
			cout<<"Saved dispersion at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving dispersion at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// JROT2
	if(which.jrot2)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Jrot2.txt";		
		output.open(temp.str(),fstream::app);
		double *jrot2=new double[which.bins];
		obs->write_jrot2(jrot2);
		for(int i=0; i<which.bins; i++)
		{
			output<<jrot2[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved jrot2 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving jrot2 at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Jrot2_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		double *jrot2=new double[which.bins];
		obs->write_jrot2(jrot2);
		for(int i=0; i<which.bins; i++)
		{
			output<<jrot2[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved jrot2 at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving jrot2 at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}

	// DRIFT 
	if(which.drift)
	{
		fstream output;
		stringstream temp;

		temp<<directory<<"/Drift.txt";		
		output.open(temp.str(),fstream::app);
		int *drift=new int[which.drift_bins];
		obs->write_drift(drift);
		for(int i=0; i<which.drift_bins; i++)
		{
			output<<drift[i]<<"\t";
		}
		output<<endl;
		if(output.good())
		{
			cout<<"Saved drift distribution at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving drift distribution at time "<<number<<" failed"<<endl;
		}
		output.close();

		/* OLD 
		temp<<directory<<"/Drift_"<<number<<".txt";		
		output.open(temp.str(),fstream::out);
		int *drift=new int[which.drift_bins];
		obs->write_drift(drift);
		for(int i=0; i<which.drift_bins; i++)
		{
			output<<drift[i]<<endl;
		}
		if(output.good())
		{
			cout<<"Saved drift distribution at time "<<number<<endl;
		}
		else
		{
			cout<<"Saving drift distribution at time "<<number<<" failed"<<endl;
		}
		output.close();
		*/
	}
}

void SaveResults::save_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int i=0; i < lattice->get_length(); i++)
	{
		output<<data1[i]<<endl;
		output<<data2[i]<<endl;
	}
	
	output.close();
}

void SaveResults::save_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int tauindex)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						output<<data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
						output<<data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
				}
			}
		}
	}
	
	if(output.good())
	{
		cout<<"Saved lattice at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving lattice at time "<<number<<" failed"<<endl;
	}
	output.close();
}

void SaveResults::save_lattice_slices(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, int min_tauindex, int max_tauindex)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/Lattice_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
	    for(int tauindex=min_tauindex; tauindex<max_tauindex; tauindex++)
		{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						output<<data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
						output<<data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]<<endl;
				}
			}
		}
		}
	}
	
	if(output.good())
	{
		cout<<"Saved lattice at time "<<number<<endl;
	}
	else
	{
		cout<<"Saving lattice at time "<<number<<" failed"<<endl;
	}
	output.close();
}



void SaveResults::read_lattice_slice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename)
{
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream input;		
	input.open(filename,fstream::in);
	complex <double> temp1,temp2;
	
	for(int compindex=0; compindex<COMPONENTS; compindex++)
	{
		for(int xindex=0; xindex < XSIZE; xindex++)
		{
			for(int yindex=0; yindex < YSIZE; yindex++)
			{
				for(int zindex=0; zindex < ZSIZE; zindex++)
				{
						input>>temp1; input>>temp2;
						for(int tauindex=0; tauindex < TAUSIZE; tauindex++)
						{
							data1[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]=temp1;
							data2[compindex*TAUSIZE*XSIZE*YSIZE*ZSIZE+tauindex*XSIZE*YSIZE*ZSIZE+xindex*YSIZE*ZSIZE+yindex*ZSIZE+zindex]=temp2;
						}
					
				}
			}
		}
	}
	input.close();
}


void SaveResults::read_lattice(ComplexLatticeHost *lattice, ComplexLatticeHost *lattice_conjug, string filename)
{
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream input;		
	input.open(filename,fstream::in);

	if (!input) { // Check if the file opened successfully
    	cerr << "File not found!" << endl;
        exit(1);
    }

	complex <double> temp1,temp2;
	
	for(int i=0; i < lattice->get_length(); i++)
	{
		
		input>>temp1; input>>temp2;
		
		data1[i] = temp1;
		data2[i] = temp2;
	}

	input.close();
}

void SaveResults::save_lattice_Spatial(ComplexLatticeSpatialHost *lattice, ComplexLatticeSpatialHost *lattice_conjug)
{
	int number=obs->get_counter();
	
	complex <double> *data1, *data2;
	data1=lattice->get_pointer();
	data2=lattice_conjug->get_pointer();
	
	fstream output;
	stringstream temp;
	temp<<directory_rawdata<<"/dipgrid_"<<number<<".txt";		
	output.open(temp.str(),fstream::out);

	//cout << lattice->get_length() << endl;
	
	for(int i=0; i < lattice->get_length(); i++)
	{
		output<<data1[i]<<endl;
		output<<data2[i]<<endl;
	}
	
	output.close();
}
