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
        print(l)
        if '349446' in l:
            print('wrong')
            continue
        
        if 'ristretto' in l:
            running = True
            break
        
    if not running:
        break
    else:
        print('still training.')

print('done training.')

cmd = 'python -m gdvb configs/dave_2x2_enu2_cov3_low3.toml gen_props 10'
cmd = 'ls'
os.system(cmd)
cmd = 'python -m gdvb configs/dave_2x2_enu2_cov3_low3.toml verify 10'
cmd = 'la'
os.system(cmd)
