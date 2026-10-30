```bash
cd /path/to/project

module load devel/cuda/12.6
module load compiler/gnu/11.3

python3 compile_script.py --mu-min 1.5 --mu-max 2.7 --mu-steps 4 --runs-per-mu 2 --card A100 && sbatch --mem=8G jobscript_Helix.sh
```

The default sweep builds two independently seeded executables for each of
`MU = 1.5, 1.9, 2.3, 2.7`. The binaries are named `bosegascl0` through
`bosegascl7`, and `build_mu_0_7.csv` records each binary's MU value and seed.
The build script updates `Parameters.h` before each compile and restores its
original contents after the sweep.
