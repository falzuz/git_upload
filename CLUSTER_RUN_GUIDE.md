```bash
cd /path/to/project

module load devel/cuda/12.6
module load compiler/gnu/11.3

python3 compile_script.py 20 0 --card A100 && sbatch --mem=8G jobscript_Helix.sh
```
