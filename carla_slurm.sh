#SBATCH --job-name {name}                    # Job name
### Logging
#SBATCH --output={log_dir}/{name}_%j.out   # Name of stdout output file (%j expands to jobId)
#SBATCH --error={log_dir}/{name}_%j.err    # Name of stderr output file (%j expands to jobId)
#SBATCH --mail-user=bzhou@cs.utexas.edu      # Email of notification
#SBATCH --mail-type=END,FAIL,REQUEUE
### Node info
###SBATCH --partition Test                   # Queue name [NOT NEEDED FOR NOW]
#SBATCH --nodes=1                            # Always set to 1 when using the cluster
#SBATCH --time 78:00:00                      # Run time (hh:mm:ss)
#SBATCH --ntasks-per-node=1                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --gres=gpu:1                         # Number of gpus needed
#SBATCH --cpus-per-task=8                    # Number of cpus needed per task
#SBATCH --mem=48G                            # Memory requirements
./train_{name}.sh
"""
