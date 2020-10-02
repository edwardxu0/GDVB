import subprocess
import time
import os

running = False
while True:
    time.sleep(10)
    process = subprocess.Popen(['squeue', '-u', 'dx3yy'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    out, err = process.communicate()
    lines = out.decode("utf-8") .split('\n')
    for l in lines:
        if 'ristretto' in l and 'R' in l:
            if int(l.split(' ')[12]) <= 352395:
                running = True
                break
            else:
                running = False
                break
        
    if not running:
        print('done training.')
        break
    else:
        print('still training.')

cmd = 'python -m gdvb configs/dave_enu2_cov3_low3.toml gen_props 10'
os.system(cmd)
cmd = 'python -m gdvb configs/dave_enu2_cov3_low3.toml verify 10'
os.system(cmd)
