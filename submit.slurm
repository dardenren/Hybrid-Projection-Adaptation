#!/bin/bash
#SBATCH --job-name=mobilebert_imdb_lora_training_1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:30:00
#SBATCH --output=output/mobilebert_output_%j.log
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1355863@u.nus.edu

# Load required modules (adjusted for compatibility)
module load python/3.10.12
module load cuda/11.2

# Activate virtual environment
source ./venv/bin/activate

# Ensure project directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR

# Force unbuffered output for real-time tqdm updates
export PYTHONUNBUFFERED=1

# Install dependencies if not already installed
pip install --no-cache-dir -r requirements.txt

# Run the main script
python main.py