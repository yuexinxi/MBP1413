#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=3  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=47000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=08:00:00     # DD-HH:MM:SS
#SBATCH --account=def-uludagk

# Prepare environment
module load python/3.11.5 cuda cudnn
module load scipy-stack

# check if they are installed
# pip install --no-index --upgrade pip
# pip install --no-index Jupyter
# pip install --no-index torch
# pip install --no-index tensorflow

# echo -e '#!/bin/bash\nexport JUPYTER_RUNTIME_DIR=$SLURM_TMPDIR/jupyter\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh

# chmod u+x $VIRTUAL_ENV/bin/notebook.sh

source $HOME/jupyter_py3/bin/activate

# Prepare data
cd $SLURM_TMPDIR
mkdir work
mkdir data
tar -xf ~/projects/def-uludagk/yuexinxi/MBP1413Data/MICCAI_BraTS2020_TrainingData.tar -C $SLURM_TMPDIR/data/
cp ~/projects/def-uludagk/yuexinxi/swin_unetr_brats20.py $SLURM_TMPDIR/work/

# Start the script
cd work
ls $SLURM_TMPDIR/data/MICCAI_BraTS2020_TrainingData/
python swin_unetr_brats20.py

# Get output from node


# Clean up
tar -cf ~/projects/def-uludagk/yuexinxi/result-archive.tar $SLURM_TMPDIR/work/
