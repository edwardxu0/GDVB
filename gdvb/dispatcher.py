import os


class Task():
    def __init__(self, cmds, dispatch, name, log_path, slurm_path):
        self.cmds = cmds
        self.mode = dispatch['mode']
        self.log_path = log_path

        if self.mode == 'slurm':
            self.slurm_path = slurm_path

            gpu = dispatch['gpu'] if 'gpu' in dispatch else False
            self.configure_slurm(self.cmds, name, gpu, log_path, slurm_path)

    # configure slurm script if running on slurm server
    def configure_slurm(self, cmds, name, gpu, log_path, slurm_path):

        lines = ['#!/bin/sh',
                 f'#SBATCH --job-name={name}',
                 f'#SBATCH --error={log_path}',
                 f'#SBATCH --output={log_path}']
        
        if gpu:
            lines += ['#SBATCH --partition=gpu',
                      '#SBATCH --gres=gpu:1']
            
        lines += ['cat /proc/sys/kernel/hostname']
        lines += cmds
        
        lines = [x+'\n' for x in lines]
        open(slurm_path, 'w').writelines(lines)


    # execute task
    def run(self):
        if self.mode == 'slurm':
            cmd = f'sbatch {self.slurm_path}'
        elif self.mode == 'local':
            for cmd in self.cmds:
                if 'dnnv' in cmd or 'r4v' in cmd:
                    cmd += f' > {self.log_path} 2>&1'
                    #cmd += f' > {self.log_path} 2>/dev/null'
                    #cmd += f' > {self.log_path} 2>{self.log_path}.err'
                #print(cmd)
                os.system(cmd)
        else:
            assert False

