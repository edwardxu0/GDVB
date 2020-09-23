#!/usr/bin/env python
import os
import sys
import argparse
import pickle
import numpy as np

from natsort import natsorted
from pathlib import Path

from gdvb.dispatcher import Task


def _parse_args():
    parser = argparse.ArgumentParser(description='E1, ACAS')
    parser.add_argument('-t','--task', type=str, choices=['verify', 'analyze'], help='task to perform')
    parser.add_argument('-d','--output_dir', type=str, default='./results/acas/', help='result director')
    parser.add_argument('-p','--platform', type=str, default='local', choices=['local','slurm'], help='')
    parser.add_argument('-nd', '--network_dir', type=str, default='./artifacts/acas_benchmark/onnx', help='acas network directory')
    parser.add_argument('-pd', '--property_dir', type=str, default='./artifacts/acas_benchmark/properties', help='acas network directory')
    parser.add_argument('-tl', '--time_limit', type=int, default=300, help='time limit')
    parser.add_argument('-ml', '--memory_limit', type=str, default='64G', help='memory limit(M/G)')
    parser.add_argument('-tpn','--task_per_node', type=int, default=7, help='number of verification jobs to run per node')
    parser.add_argument('-n','--nodes', type=str, nargs='+',
                        default=['cortado02','cortado03', 'cortado04', 'cortado05',
                                 'cortado06', 'cortado07','cortado08', 'cortado09', 'cortado10'],
                            help='a list of nodes to run verification')
    parser.add_argument('-v','--verifiers', type=str, nargs='+',
                        default=['eran_deepzono','eran_deeppoly','eran_refinezono','eran_refinepoly',
                                 'neurify','planet','bab','bab_sb','reluplex'],
                        help='a list of verification tools')
    
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.task == 'verify':
        verify(args)
    elif args.task == 'analyze':
        analyze(args)
    else:
        assert False

        
def verify(args):
    net_dir = args.network_dir
    prop_dir = args.property_dir
    slurm_dir = os.path.join(args.output_dir,"veri_slurm")
    log_dir = os.path.join(args.output_dir,"veri_log")
    Path(slurm_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    verifiers = args.verifiers
    nets = natsorted(os.listdir(net_dir))
    props = natsorted(os.listdir(prop_dir))

    setup = acas_verification_setup()

    dispatch_args = {}
    dispatch_args['platform'] = args.platform
    dispatch_args['nodes'] = args.nodes
    dispatch_args['task_per_node'] = args.task_per_node
    

    c = 0
    for v in verifiers:
        if 'eran' in v:
            v_name = 'eran'
            verifier_parameters = f'--eran.domain {v.split("_")[1]}'
        elif v == 'bab_sb':
            v_name = 'bab'
            verifier_parameters = '--bab.smart_branching True'
        else:
            v_name = v
            verifier_parameters = ""

        for p in props:
            for n in nets:
                p_name = int(p.split('_')[1].split('.')[0])
                n_name = (int(n.split('_')[1]), int(n.split('_')[2].split('.')[0]))

                if not n_name in setup[p_name-1]:
                    continue

                c+=1
                print(f'{c}/{len(verifiers)*186}')
                
                cmd = f'python -W ignore ./lib/DNNV/tools/resmonitor.py -T {args.time_limit} -M {args.memory_limit}'
                cmd += f' ./scripts/run_dnnv.sh {os.path.join(prop_dir,p)} --network N  {os.path.join(net_dir,n)} --{v_name} {verifier_parameters}'

                cmds = [cmd]
                task = Task(cmds,
                            dispatch_args,
                            "GDVB_Verify",
                            os.path.join(log_dir,f'{v}_{p_name}_{n_name[0]}x{n_name[1]}.out'),
                            os.path.join(slurm_dir,f'{v}_{p_name}_{n_name[0]}x{n_name[1]}.slurm')
                )
                task.run()


def analyze(args):
    log_dir = os.path.join(args.output_dir,"veri_log")
    results = {}

    logs = sorted(os.listdir(log_dir))

    for log in logs:
        v = log.split('_')[0]
        if v == 'eran':
            v += '_'+log.split('_')[1]
        elif v == 'bab':
            if log.split('_')[1] == 'sb':
                v += '_sb'
            
        log = os.path.join(log_dir, log)
        rlines = [x for x in reversed(open(log,'r').readlines())]
        
        res = None
        v_time = None

        for i,l in enumerate(rlines):
            if l.startswith('INFO'):# or l.startswith('DEBUG'):
                continue 

            if '[STDERR]:Error: GLP returned error 5 (GLP_EFAIL)' in l:
                res = 'error'
                #print(log)
                break

            if "*** Error in `python':" in l:
                res = 'error'
                #print(log)
                break

            if 'Cannot serialize protocol buffer of type ' in l:
                res = 'error'
                #print(log)
                break

            if 'OverflowError: integer division res' in l:
                res = 'error'
                break
            
            if 'Error' in l:
                res = 'error'
                #print(log)
                break

            if 'Timeout' in l:
                res = 'timeout'
                break

            elif 'Out of Memory' in l:
                res = 'memout'
                break

            if l.startswith('  result: '):
                if 'Unsupported' in l or 'not support' in l or 'Unknown MIPVerify' in l or 'Unknown property check result' in l:
                    res = 'unsup'
                    break
                elif 'NeurifyError' in l or 'PlanetError' in l:
                    res = 'error'
                    break
                else:
                    res = l.strip().split(' ')[-1]
                    v_time = float(rlines[i-1].strip().split(' ')[-1])
                    break

            # remove this
            if i == len(rlines)-1:
                res = 'running'
                #print(res, log)

        if res not in ['sat','unsat']:
            v_time = args.time_limit

        assert res in ['sat','unsat','unknown','timeout','memout','error', 'unsup', 'running', 'torun'], (log,res)
        if not v in results:
            results[v] = [(res,v_time)]
        else:
            results[v] += [(res,v_time)]

    with open(f'{args.output_dir}.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    # calculate scr and par2 scores
    scr_dict = {}
    par_2_dict = {}
    for v in args.verifiers:
        sums = []
        scr = 0
        for r in results[v]:
            res = r[0]
            vtime = r[1]
            if res == "running":
                pass
                #print(f"running: ")
            if res in ["sat", "unsat"]:
                scr += 1
                sums += [vtime]
            elif res in [
                "unknown",
                "timeout",
                "memout",
                "error",
                "unsup",
                "running",
            ]:
                sums += [args.time_limit * 2]
            elif res in ["untrain"]:
                pass
            else:
                assert False, res

        par_2 = np.mean(np.array(sums))

        if v not in scr_dict.keys():
            scr_dict[v] = [scr]
            par_2_dict[v] = [par_2]
        else:
            scr_dict[v] += [scr]
            par_2_dict[v] += [par_2]


    print('{:>15}  {:>15}  {:>15}'.format('Verifier','SCR','PAR-2'))
    print('-------------------------------------------------')
    for v in args.verifiers:
        print('{:>15}  {:>15}  {:>15}'.format(v, scr_dict[v][0], par_2_dict[v][0]))


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
