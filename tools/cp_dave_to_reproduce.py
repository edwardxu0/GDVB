#!/usr/bin/env python

import os
import numpy as np
from decimal import Decimal as D
from shutil import copyfile

cav_model_path = '/bigtemp/dx3yy/gdvb_cav/dave.11/dis_log'
new_model_path = '/p/d4v/dx3yy/Work/GDVB/res_ase/s1_dave2.10/dis_log'
cav_models = sorted(os.listdir(cav_model_path))
new_models_attic = sorted(os.listdir(new_model_path+'_new'))

factor_order = ['neu','fc','conv','idm','ids']
factor_levels = {
    'neu'  : np.arange(D('0.2'),D('1.2'),D('0.2')),
    'fc'   : np.array([D('0.0'),D('0.25'),D('0.5'),D('0.75'),D('1.0')]),
    'conv' : np.arange(D('0.0'),D('1.2'),D('0.2')),
    'idm'  : np.arange(D('0.2'),D('1.2'),D('0.2')),
    'ids'  : np.arange(D('0.2'),D('1.2'),D('0.2'))
}

print(cav_models)
print(new_models_attic)
print(factor_levels)

for cm in cav_models:
    print(cm)
    factor_indexs = cm.split('_')[0][5:]
    post_fix = cm.split('_')[1]

    new_name = ''
    for i, fi in enumerate(factor_indexs):
        factor = factor_order[i]

        new_name += f'{factor}={factor_levels[factor][int(fi)]}_'
    new_name += 'SF='+post_fix

    assert new_name in new_models_attic, f"original model: {new_name} doesn't exist!"

    cmd = copyfile(os.path.join(cav_model_path,cm), os.path.join(new_model_path,new_name))
