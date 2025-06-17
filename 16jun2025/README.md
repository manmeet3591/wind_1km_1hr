ibrun -np 16 ./run_swin_parallel.sh c611-102 16

apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv apptainer_wind.sif python swin_transformer_wind_operational.py
