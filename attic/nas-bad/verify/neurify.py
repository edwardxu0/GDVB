import os
import re


def analyze_result(path):
    lines = open(path, 'r').readlines()

    time = -1
    prop_res = None

    for l in lines:
        if 'Segmentation fault' in l or 'Killed' in l:
            prop_res = 'Error'

        elif 'CANCELLED' in l:
            prop_res = 'CANCELED'
            break

        elif l.startswith("ERROR"):
            if 'Out of Memory' in l:
                prop_res = 'MemOut'
            elif 'Timeout' in l:
                prop_res = 'TimeOut'
            else:
                assert False

        elif 'Result:' in l:
            if 'Avd found!' in l:
                prop_res = 'False'
            elif 'No adv' in l:
                prop_res = 'True'
            elif "Can't prove!" in l:
                prop_res = 'Unknown'
            else:
                assert False

        elif 'real\t' in l:
            tokens = re.split('m|s', l[5:])
            time = float(tokens[0]) * 60 + float(tokens[1])
            done = True

    return([prop_res, time])
