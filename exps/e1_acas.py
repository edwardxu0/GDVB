#!/usr/bin/env python
import os
import sys
import argparse

from natsort import natsorted
from pathlib import Path

from gdvb.dispatcher import Task


def _parse_args():
    parser = argparse.ArgumentParser(description='E1, ACAS')
    parser.add_argument('-d','--output_dir', type=str, default='./results/acas/', help='result director')
    parser.add_argument('-p','--platform', type=str, default='local', choices=['local','slurm'], help='')
    parser.add_argument('-nd', '--network_dir', type=str, default='./configs/networks/acas', help='acas network directory')
    parser.add_argument('-t', '--time_limit', type=int, default=300, help='time limit')
    parser.add_argument('-m', '--memory_limit', type=str, default='64G', help='memory limit(M/G)')
    parser.add_argument('-v','--verifiers', type=str, nargs='+',
                        default=['eran_deepzono','eran_deeppoly','eran_refinezono','eran_refinepoly',
                                 'neurify','planet','bab','bab_sb','reluplex'],
                        help='a list of verification tools')
    
    return parser.parse_args()


def main():
    args = _parse_args()

    gen_props(args)
    verify(args)


def gen_props(args):
    cmd = f'./tools/generate_acas_properties.py -d {os.path.join(args.output_dir,"props")}'
    os.system(cmd)


def verify(args):
    net_dir = args.network_dir
    prop_dir = os.path.join(args.output_dir,"props")
    slurm_dir = os.path.join(args.output_dir,"veri_slurm")
    log_dir = os.path.join(args.output_dir,"veri_log")
    Path(slurm_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    verifiers = args.verifiers
    nets = natsorted(os.listdir(net_dir))
    props = natsorted(os.listdir(prop_dir))

    setup = acas_verification_setup()

    dispatch_args = {}
    dispatch_args['mode'] = args.platform

    c = 0
    for v in verifiers:
        if 'eran' in v:
            v_name = 'eran'
            verifier_parameters = f'--eran.domain {v.split("_")[1]}'
        elif v == 'bab_sb':
            v_name = 'bab'
            verifier_parameters = '--bab.smart_branching'
        else:
            v_name = v
            verifier_parameters = ""

        for p in props:
            for n in nets:
                p_name = int(p.split('_')[0])
                n_name = (int(n.split('_')[1]), int(n.split('_')[2].split('.')[0]))

                if n_name not in setup[p_name-1]:
                    pass

                c+=1
                
                cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {args.time_limit} -M {args.memory_limit}'
                cmd += f' python -m dnnv {os.path.join(net_dir,n)} {os.path.join(prop_dir,p)} --{v_name} {verifier_parameters}'

                cmds = [cmd]
                task = Task(cmds,
                            dispatch_args,
                            "GDVB_Verify",
                            os.path.join(log_dir,f'{v}_{p_name}_{n_name[0]}x{n_name[1]}.out'),
                            os.path.join(slurm_dir,f'{v}_{p_name}_{n_name[0]}x{n_name[1]}.slurm')
                )
                task.run()


def acas_verification_setup():
    p = []
    for x in range(1,6):
        for y in range(1,10):
            p += [(x,y)]
    setup = [p,p,p,p] # p1-4
    setup += [[(1,1)]] # p5
    setup += [[(1,1)]] # p6
    setup += [[(1,9)]] # p7
    setup += [[(2,9)]] # p8
    setup += [[(3,3)]] # p9
    setup += [[(4,5)]] # p10
    return setup


if __name__ == '__main__':
    main()
