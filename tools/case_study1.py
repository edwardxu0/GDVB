#!/usr/bin/env python
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser(description='CS1', prog='CS1')
parser.add_argument('--name', type=str)
parser.add_argument('--root', type=str, default='./results',help='root directory')
args = parser.parse_args()


def get_res(name, seeds, verifiers):
    scr_dict = {}
    par_2_dict = {}

    for seed in seeds:
        with open(f'{args.root}/verification_results_{name}_{seed}.pickle', 'rb') as handle:
            results = pickle.load(handle)

        for v in verifiers:
            sums = []
            scr = 0
            for vpc_k in results[v]:
                res = results[v][vpc_k][0]
                vtime = results[v][vpc_k][1]

                if res in ['sat','unsat']:
                    scr += 1
                    sums += [vtime]
                elif res in ['unknown','timeout','memout','error','unsup']:
                    sums += [14400*2]
                elif res in ['untrain']:
                    pass
                else:
                    assert False, res

            par_2 = int(round(np.mean(np.array(sums))))

            if v not in scr_dict.keys():
                scr_dict[v] = [scr]
                par_2_dict[v] = [par_2]
            else:
                scr_dict[v] += [scr]
                par_2_dict[v] += [par_2]

    return scr_dict, par_2_dict, len(results[v])


seeds = [10,11,12,13,14]
verifiers = ['eran_deepzono','eran_deeppoly','eran_refinezono','eran_refinepoly','neurify','planet','bab','bab_sb','mipverify','reluplex']

scr_dict_mcb, par_2_dict_mcb, nb_tm = get_res('mcb', seeds, verifiers)
scr_dict_dave, par_2_dict_dave, nb_td = get_res('dave', seeds, verifiers)

print(nb_td, nb_tm)

#print('| Verifier | SCR | PAR-2 |')
#print('|---|---|---|')

#print('\\begin{tabular}{|c|c|c|c|c|c|c|}')
#print('\\hline')
#print(' & \\multicolumn{3}{|c|}{MNIST} &\\multicolumn{3}{|c|}{DAVE-2} \\\\')
#print('\\hline')
for v in verifiers:
    sm = scr_dict_mcb[v]
    pm = par_2_dict_mcb[v]
    sd = scr_dict_dave[v]
    pd = par_2_dict_dave[v]
    spm1 = np.mean(sm)/nb_tm*100
    spm2 = np.std(sm)/nb_tm*100
    
    spd1 = np.mean(sd)/nb_td*100
    spd2 = np.std(sd)/nb_td*100
    
    #print(f'|{v}|{np.mean(sm)}+-{np.std(sm):.2f}|{np.mean(pm)}+-{np.std(pm):.2f}|')
    v2 = v.replace("_","\\_")
    print(f'{v2} & {np.mean(sm)}$\\pm${np.std(sm):.2f} & {spm1:.2f}\\%$\\pm${spm2:.2f}\\% & {np.mean(pm)}$\\pm${np.std(pm):.2f}& {np.mean(sd)}$\\pm${np.std(sd):.2f} & {spd1:.2f}\\%$\\pm${spd2:.2f}\\% & {np.mean(pd)}$\\pm${np.std(pd):.2f}\\\\')
#print('\\hline')
#print('\\end{tabular}')
