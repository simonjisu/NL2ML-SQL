#!/bin/bash

#SBATCH --job-name=grpo_training                    # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:4                          # Using 1 gpu
#SBATCH --time=0-12:00:00                     # 1 hour timelimit
#SBATCH --mem=25000MB                         # Using 10GB CPU Memory
#SBATCH --partition=A                         # Using "A" partition
#SBATCH --cpus-per-task=4                     # Using 4 maximum processor
#SBATCH --signal=B:SIGUSR1@30                 # Send SIGUSR1 within 30s of its end time
#SBATCH --output=slurm_log/%x-%j.out          # Save standard output file (%x=job name, %j=job id)
#SBATCH --open-mode=append                    # Open the output and error files using append mode

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate {conda sql}

max_restarts=2
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*****' | cut -d= -f2)

function resubmit()
{
    if [[ $restarts -lt $max_restarts ]]; then
        scontrol requeue ${SLURM_JOB_ID}
        exit 0
    else
        echo "Your job is over the Maximum restarts limit"
        exit 1
    fi
}

trap 'resubmit' SIGUSR1

accelerate launch --config_file config.yaml grpo_multi.py &
wait
exit 0