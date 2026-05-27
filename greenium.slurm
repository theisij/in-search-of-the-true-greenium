#!/usr/bin/env bash
#
#SBATCH --job-name=greenium                # Job name
#SBATCH --output=logs/greenium-%j.out      # stdout (%j = jobid)
#SBATCH --error=logs/greenium-%j.err       # stderr
#SBATCH --ntasks=1                         # one task (no MPI)
#SBATCH --cpus-per-task=48                 # threads per task
#SBATCH --mem=300G                         # RAM
#SBATCH --time=24:00:00                    # HH:MM:SS
#SBATCH --partition=gpunormal                 # Cluster partition
#SBATCH --mail-type=ALL                    # Send email on start, end and fail
#SBATCH --mail-user=theis.jensen@yale.edu


echo '-------------------------------'
cd ${SLURM_SUBMIT_DIR}                         # Change directory to slurm submit directory
echo ${SLURM_SUBMIT_DIR}
echo Running on host $(hostname)
echo Time is $(date)
echo SLURM_NODES are ${SLURM_NODELIST}
echo '-------------------------------'
echo -e '\n\n'

# # Reproducible Python env via uv
uv sync                   # installs Python & dependencies into .venv/
source .venv/bin/activate

# # Run the Python pipeline
echo "Running Python main.py..."
python -u -m greenium.main

# Create conda environment once 
# conda create -n greenium -c conda-forge r-base r-tidyverse r-data.table r-arrow r-lfe r-broom
# conda create -n greenium -c conda-forge r-base=4.5.0 r-tidyverse=2.0.0 r-data.table=1.17.0 r-arrow=19.0.1.1 r-lfe=3.1.1 r-broom=1.0.8 -y

# Setup conda environment to run R script
module load anaconda3/2023.09-0-k3at
source "$(conda info --base)/etc/profile.d/conda.sh" # To make the conda command available
conda activate greenium

# Run the R script
echo "Running R script..."
Rscript R/main.R
conda deactivate

echo "Finished all tasks."