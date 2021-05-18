import os
import subprocess
import random
import string
import time
import re


class Task():
    def __init__(self, cmds, dispatch, name, log_path, slurm_path):
        self.cmds = cmds
        self.platform = dispatch['platform']
        self.log_path = log_path
        self.reservation = dispatch['reservation'] if 'reservation' in dispatch else None
        self.nb_cpus = dispatch['nb_cpus'] if 'nb_cpus' in dispatch else None
        self.exclude = dispatch['exclude'] if 'exclude' in dispatch else None

        if self.platform == 'slurm':
            self.slurm_path = slurm_path

            gpu = dispatch['gpu'] if 'gpu' in dispatch else None
            self.configure_slurm(self.cmds, name, gpu, log_path, slurm_path, dispatch)

    # configure slurm script if running on slurm server
    def configure_slurm(self, cmds, name, gpu, log_path, slurm_path, dispatch):
        self.nodes = dispatch['nodes'] if 'nodes' in dispatch else None
        self.task_per_node = dispatch['task_per_node'] if 'task_per_node' in dispatch else None
        
        lines = ['#!/bin/sh',
                 f'#SBATCH --job-name={name}',
                 f'#SBATCH --error={log_path[:-4]}.err',
                 f'#SBATCH --output={log_path}']
        if gpu:
            lines += ['#SBATCH --partition=gpu',
                      '#SBATCH --gres=gpu:1']

        lines += ['export GRB_LICENSE_FILE=/u/dx3yy/.gurobikeys/`hostname`.gurobi.lic'] # hard coded gurobi license
        lines += ['cat /proc/sys/kernel/hostname']
        lines += cmds
        
        lines = [x+'\n' for x in lines]
        open(slurm_path, 'w').writelines(lines)


    # execute task
    def run(self):
        if self.platform == 'slurm':
            cmd = 'sbatch'
            cmd += f' --exclude={self.exclude}' if self.exclude else ''
            cmd += f' --reservation {self.reservation}' if self.reservation else ''
            cmd += f' -c {self.nb_cpus}' if self.nb_cpus else ''

            if self.nodes:
                print(f'Requesting a node from: {self.nodes}')
                node = self.request_node()
                cmd += f' -w {node}'
            cmd +=  f' {self.slurm_path}'
        elif self.platform == 'local':
            for cmd in self.cmds:
                if 'dnnv' in cmd or 'r4v' in cmd:
                    cmd += f' > {self.log_path} 2>&1'
                    #cmd += f' > {self.log_path} 2>/dev/null'
                    #cmd += f' > {self.log_path} 2>{self.log_path}.err'
        else:
            assert False

        subprocess.call(cmd, shell=True)
    
        
    def request_node(self):
        while(True):
            node_avl_flag = False
            tmp_file = './tmp/'+''.join(random.choice(string.ascii_lowercase) for i in range(16))
            
            #sqcmd = f'squeue | grep cortado > {tmp_file}'
            sqcmd = f'squeue  > {tmp_file}'
            #sqcmd = f'squeue -u dx3yy > {tmp_file}'
            time.sleep(3)
            os.system(sqcmd)
            sq_lines = open(tmp_file, 'r').readlines()[1:]
            os.remove(tmp_file)

            nodes_avl = {}
            for node in self.nodes:
                nodes_avl[node] = 0

            nodenodavil_flag = False
            for l in sq_lines:
                if 'ReqNodeNotAvail' in l and 'dx3yy' in l:
                    nodenodavil_flag = True
                    unavil_node = l[:-1].split(',')[-1].split(':')[1][:-1]
                    if unavil_node in self.nodes:
                        cmd = f'sinfo > tmp/sinfo.txt'
                        os.system(cmd)
                        sinfo_lines = open('tmp/sinfo.txt', 'r').readlines()
                        
                        temp = re.compile("([a-zA-Z]+)([0-9]+)") 
                        nname, digits = temp.match(unavil_node).groups()
                        for sl in sinfo_lines:
                            if 'drain' in sl or 'down' in sl or 'drng' in sl:
                                if nname in sl and digits in sl:
                                    self.nodes.remove(unavil_node)
                                    if len(self.nodes) == 0:
                                        print('No avilable nodes. exiting.')
                                        exit()
                                    nodenodavil_flag = False
                    else:
                        nodenodavil_flag = False
                    if nodenodavil_flag:
                        break

                if ' R ' in l and l != '':
                    node = l.strip().split(' ')[-1]
                    if node in self.nodes:
                        nodes_avl[node] += 1

            if nodenodavil_flag:
                print('node unavialiable. waiting ...')
                continue

            #print(nodes_avl)

            for na in nodes_avl:
                if nodes_avl[na] < self.task_per_node:
                    node_avl_flag = True
                    break
            if node_avl_flag:
                break
    
        return na
