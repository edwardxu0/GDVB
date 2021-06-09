import os
import subprocess
import random
import string
import time
import re


class Task:
    def __init__(self, cmds, dispatch, task_name, output_path, slurm_path):
        self.cmds = cmds
        self.platform = dispatch['platform']
        self.output_path = output_path
        self.error_path = f'{os.path.splitext(output_path)[0]}.err'

        self.exclude = dispatch['exclude'] if 'exclude' in dispatch else None
        self.reservation = dispatch['reservation'] if 'reservation' in dispatch else None
        self.nb_cpus = dispatch['nb_cpus'] if 'nb_cpus' in dispatch else None
        self.nodes = dispatch['nodes'] if 'nodes' in dispatch else None
        self.task_per_node = dispatch['task_per_node'] if 'task_per_node' in dispatch else None

        if self.platform == 'slurm':
            self.slurm_path = slurm_path
            nb_gpus = dispatch['nb_gpus'] if 'nb_gpus' in dispatch else 0
            self.configure_slurm(self.cmds, task_name, nb_gpus)

    # configure slurm script if running on slurm server
    def configure_slurm(self, cmds, task_name, nb_gpus):
        lines = ['#!/bin/sh',
                 f'#SBATCH --job-name={task_name}',
                 f'#SBATCH --output={self.output_path}',
                 f'#SBATCH --error={self.error_path}']
        if nb_gpus != 0:
            lines += ['#SBATCH --partition=gpu',
                      f'#SBATCH --gres=gpu:{nb_gpus}']

        # TODO: remove hard coded gurobi license
        lines += ['export GRB_LICENSE_FILE=/u/dx3yy/.gurobikeys/`hostname`.gurobi.lic']
        lines += ['cat /proc/sys/kernel/hostname']
        lines += cmds
        
        lines = [x+'\n' for x in lines]
        open(self.slurm_path, 'w').writelines(lines)

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
            cmd += f' {self.slurm_path}'
            subprocess.call(cmd, shell=True)

        elif self.platform == 'local':
            for cmd in self.cmds:
                cmd += f' > {self.output_path} 2> {self.error_path}'
                subprocess.call(cmd, shell=True)
        else:
            assert False

    def request_node(self):
        while True:
            node_avl_flag = False
            tmp_file = './tmp/'+''.join(random.choice(string.ascii_lowercase) for i in range(16))
            
            # sqcmd = f'squeue | grep cortado > {tmp_file}'
            sqcmd = f'squeue  > {tmp_file}'
            # sqcmd = f'squeue -u dx3yy > {tmp_file}'
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

            # print(nodes_avl)

            for na in nodes_avl:
                if nodes_avl[na] < self.task_per_node:
                    node_avl_flag = True
                    break
            if node_avl_flag:
                break
    
        return na
