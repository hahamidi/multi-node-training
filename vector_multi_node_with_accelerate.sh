#!/bin/bash
#SBATCH --job-name=multiple-nodes-multiple-gpus
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1  # This matches the number of GPUs per node
#SBATCH --cpus-per-task=30
#SBATCH --qos=m3
#SBATCH --partition=a40
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00
#SBATCH --mem=64gb
#SBATCH --output=./slurm_%j.out
#SBATCH --error=./slurm_%j.err
#SBATCH --comment "Key=Monitoring,Value=ON"
#SBATCH --exclude=gpu172,gpu166,gpu144,gpu138,gpu177


source ~/.bashrc
conda activate ldm2

export GPUS_PER_NODE=4
######################

######################
#### Set network #####
######################

export MASTER_ADDR=$(hostname --fqdn)
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

######################

export LAUNCHER="accelerate launch \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --rdzv_backend c10d \
    --main_process_ip $MASTER_ADDR \
    --main_process_port 29500 \
    --mixed_precision fp16 \
    "

echo $LAUNCHER
# Path to the Python script
export SCRIPT="/fs01/home/hhamidi/diffusers/examples/text_to_image_medical/text_to_image_from_org/TTIM_org.py"




  export SCRIPT_ARGS='--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
  --dataset_name="mimic-cxr" \
  --train_batch_size=2 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=100000 \
  --learning_rate=5e-05 \
  --use_ema \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --output_dir="/checkpoint/hhamidi/12833332" \
  --report_to="all" \
  --validation_prompts="" \
  --resume_from_checkpoint="latest"\
  '

CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"

for index in $(seq 0 $(($SLURM_NTASKS-1))); do 
    /opt/slurm/bin/srun -lN$index --mem=64G --gres=gpu:4 -c $SLURM_CPUS_ON_NODE -N 1 -n 1 -r $index bash -c "$CMD" &
done

wait