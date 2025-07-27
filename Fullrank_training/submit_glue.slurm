#!/bin/bash
#SBATCH --job-name=fullrank_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=02:30:00
#SBATCH --output=output/output_%j.log
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=e1355863@u.nus.edu,e1426563@u.nus.edu

# Valid tasks for GLUE
# VALID_TASKS= ("cola" "sst2" "mrpc" "stsb" "qqp" "mnli" "qnli" "rte" "wnli")

#Default values for arguments
dataset_name=""
task_name=""
model_name=""
lr=""
train_batch_size=""
epochs=""

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
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done


# Check if required arguments are provided
# task_name is optional
if [ -z "$dataset_name" ] || [ -z "$model_name" ] || [ -z "$lr" ] || [ -z "$train_batch_size" ] || [ -z "$epochs" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: sbatch submit.slurm --dataset_name <name> --model_name <name> --lr <value> --train_batch_size <value> --epochs <value>"
    exit 1
fi

# Validate GLUE task
# if ! contains "$task_name" "${VALID_TASKS[@]}"; then
#     echo "Error: Invalid task '$task_name'. Valid tasks: ${VALID_TASKS[*]}"
#     exit 1
# fi

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


# echo "Running GLUE task: $TASK, model: $MODEL"
# Example command using Hugging Face Transformers (adjust as needed)
python run_glue.py \
    --dataset_name "$dataset_name" \
    --task_name "$task_name" \
    --model_name "$model_name" \
    --lr "$lr" \
    --train_batch_size "$train_batch_size" \
    --epochs "$epochs"