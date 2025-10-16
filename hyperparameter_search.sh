#!/bin/bash
# Fixed values
fixed_hi=1024
fixed_enNL=5
fixed_fcNL=3


# Vary hidden size
for hi in 256 512 1024 2048; do
  job_name="ZNN_hi${hi}"
  script_name="slurm_${job_name}.sh"

  cat <<EOF > "$script_name"
#!/bin/bash
#SBATCH --exclude=pool32
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -J "$job_name"
#SBATCH --mail-user=haertelk@physik.hu-berlin.de
#SBATCH --time=47:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py \\
    -v 3006_1041_Z_TrE_4to8sparse_nz40_new_hi${hi}_en${enNL}_fc${fcNL} \\
    -e 9000 -nA 9 -nz 40 -m DQN -l labels_only_Dist.csv \\
    -fa _4to8s_-50def_15B_new_40Z -hi $hi  -enNL $fixed_enNL -fcNL $fixed_fcNL
EOF

  sbatch "$script_name"
done

# Vary encoder layers
for enNL in {2..8}; do
  job_name="ZNN_en${enNL}"
  script_name="slurm_${job_name}.sh"

  cat <<EOF > "$script_name"
#!/bin/bash
#SBATCH --exclude=pool32
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -J "$job_name"
#SBATCH --mail-user=haertelk@physik.hu-berlin.de
#SBATCH --time=47:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py \\
    -v 3006_1041_Z_TrE_4to8sparse_nz40_new_hi${hi}_en${enNL}_fc${fcNL} \\
    -e 9000 -nA 9 -nz 40 -m DQN -l labels_only_Dist.csv \\
    -fa _4to8s_-50def_15B_new_40Z -hi $fixed_hi -enNL $enNL -fcNL $fixed_fcNL
EOF

  sbatch "$script_name"
done

# Vary fully connected layers
for fcNL in 3 4 5; do
  job_name="ZNN_fc${fcNL}"
  script_name="slurm_${job_name}.sh"

  cat <<EOF > "$script_name"
#!/bin/bash
#SBATCH --exclude=pool32
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH -J "$job_name"
#SBATCH --mail-user=haertelk@physik.hu-berlin.de
#SBATCH --time=47:00:00
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export PYTHONFAULTHANDLER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/data/scratch/haertelk/Masterarbeit/python-venv/bin/python -u lightningTrain.py \\
    -v 3006_1041_Z_TrE_4to8sparse_nz40_new_hi${hi}_en${enNL}_fc${fcNL} \\
    -e 9000 -nA 9 -nz 40 -m DQN -l labels_only_Dist.csv \\
    -fa _4to8s_-50def_15B_new_40Z -hi $fixed_hi -enNL $fixed_enNL -fcNL $fcNL
EOF

  sbatch "$script_name"
done