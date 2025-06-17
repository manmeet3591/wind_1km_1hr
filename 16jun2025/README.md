ibrun -np 16 ./run_swin_parallel.sh c611-102 16

apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv apptainer_wind.sif python swin_transformer_wind_operational.py

idev -A ATM23017 -p gh -N 16 -n 16 -t 6:00:00
