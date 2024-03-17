#!/bin/bash

#SBATCH --account=cocoflops
#SBATCH --partition=cocoflops
#SBATCH --nodes=1
#SBATCH --nodelist=cocoflops-hgx-1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --time=7-0
#SBATCH --output=job_output.%j.out
#SBATCH --error=job_output.%j.err


if [ "$(hostname)" = "cocoflops1.stanford.edu" ] || [ "$(hostname)" = "cocoflops2.stanford.edu" ]; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/scr/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/scr/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/scr/miniconda3/etc/profile.d/conda.sh"  # commented out by conda initialize
        else
            export PATH="/scr/miniconda3/bin:$PATH"  # commented out by conda initialize
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
fi

if [ "$(hostname)" = "cocoflops-hgx-1" ]; then
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/scr/kanishkg/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/scr/kanishkg/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/scr/kanishkg/miniconda3/etc/profile.d/conda.sh"  # commented out by conda initialize
        else
            export PATH="/scr/kanishkg/miniconda3/bin:$PATH"  # commented out by conda initialize
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
fi

conda activate apa

cd ~/RLHF-APA/examples
python eval.py --pt /scr/kanishkg/trl/outputs_2/checkpoint_04000/pytorch_model/mp_rank_00_model_states.pt -n 1000 -o 0 -b 32

