#!/usr/bin/env python
import sys
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


log_path = sys.argv[1]

if 'acas' in log_path:
    artifact = 'acas'
    time_lim = 300
    nb_instance = 186
elif 'dave' in log_path:
    artifact = 'dave'
    time_lim = 14400
    nb_instance = 28
else:
    assert False
    
log = pickle.load(open(sys.argv[1],'rb'))
fig = go.Figure()


table = np.chararray((nb_instance, len(log)))
table = [[0]*len(log)]*nb_instance

for i,v in enumerate(log):
    solved = []
    for j,vpc in enumerate(log[v]):
        if artifact == 'acas':
            r = vpc
        elif artifact == 'dave':
            r = log[v][vpc]
        else:
            assert False
        if r[0] in ['sat','unsat']:
            solved += [r[1]]
        table[j][i] = (r[0],r[1])
    solved = sorted(solved)
    print(solved)
    fig.add_trace(go.Scatter(x=list(range(len(solved))), y=solved, mode='lines', name = v))


line = '|Property/Verifier|'
for v in log:
    line += f'{v} |'

for i in range(nb_instance):
    line = f'|{i}|'
    for j in range(len(log)):
        line += f'{table[i][j]} |'
    print(line)
print(table)

fig.update_layout(
    xaxis_title="Number of Instances Verified",
    yaxis_title="Time(s)",
    xaxis=dict(range=[0, nb_instance]), yaxis=dict(range=[0, time_lim]))

fig.show()
#fig.write_image('cactus.png')
