#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=128
#SBATCH --gres=gpu:A100-SXM4:4
#SBATCH --time=15:30:00
#SBATCH --partition=testp
#SBATCH -error=error.%J.err
#SBATCH --output=output.%J.out

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

export http_proxy=http://proxy-10g.10g.siddhi.param:9090
export https_proxy=http://proxy-10g.10g.siddhi.param:9090
export ftp_proxy=http://proxy-10g.10g.siddhi.param:9090

export WANDB_API_KEY='4883a15d69990032fd28ba66b983caf542ea78f5'

cd /nlsasfs/home/dialogue/abhisekt/Anisha/MedQA/Pytorch_ECAI
source /nlsasfs/home/dialogue/abhisekt/anaconda3/etc/profile.d/conda.sh
conda activate multisum

mkdir -p logs

python3 trial1-img.py > logs/output-img.log 2>&1

echo "Job finished at $(date)"