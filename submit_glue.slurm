#!/bin/bash
#SBATCH --job-name={model_name}_{dataset_name}_{task_name}_{training_type}_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:30:00
#SBATCH --output=output/{model_name}_{dataset_name}_{task_name}_{training_type}_%j.log
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1355863@u.nus.edu,e1426563@u.nus.edu

# Valid tasks for GLUE
# "cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli"

# Default values for arguments
dataset_name=""
task_name=""
model_name=""
lr=""
train_batch_size=""
epochs=""
training_type=""

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_name)
            dataset_name="$2"
            shift 2
            ;;
        --task_name)
            task_name="$2"
            shift 2
            ;;
        --model_name)
            model_name="$2"
            shift 2
            ;;
        --lr)
            lr="$2"
            shift 2
            ;;
        --train_batch_size)
            train_batch_size="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --training_type)
            training_type="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$dataset_name" ] || [ -z "$model_name" ] || [ -z "$lr" ] || [ -z "$train_batch_size" ] || [ -z "$epochs" ] || [ -z "$training_type" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch run_glue.slurm --dataset_name <name> --model_name <name> --lr <value> --train_batch_size <value> --epochs <value> --training_type <fullrank|galore|lora|hpa>"
    exit 1
fi

# Map training_type to directory name
case "$training_type" in
    fullrank)
        training_dir="Fullrank_training"
        ;;
    galore)
        training_dir="Galore_training"
        ;;
    lora)
        training_dir="LoRA_training"
        ;;
    hpa)
        training_dir="HPA_training"
        ;;
    *)
        echo "Error: Invalid training_type. Must be one of: fullrank, galore, lora, hpa"
        exit 1
        ;;
esac

# Load required modules
module load python/3.10.12
module load cuda/11.2

# Install dependencies if not already installed
pip install --no-cache-dir -r requirements.txt

pip install --no-cache-dir tf-keras scikit-learn
pip uninstall -y tensorflow

# Change to the appropriate training directory
cd "$training_dir" || { echo "Error: Cannot change to directory $training_dir"; exit 1; }

# Activate virtual environment
source ./venv/bin/activate

# Ensure project directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR/$training_dir

# Force unbuffered output for real-time tqdm updates
export PYTHONUNBUFFERED=1


# Run the GLUE task
python run_glue.py \
    --dataset_name "$dataset_name" \
    --task_name "$task_name" \
    --model_name "$model_name" \
    --lr "$lr" \
    --train_batch_size "$train_batch_size" \
    --epochs "$epochs"