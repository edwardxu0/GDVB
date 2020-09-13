#!/usr/bin/env python
import sys
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


log = pickle.load(open(sys.argv[1],'rb'))

verifier_names = [x for x in log]
#fig = make_subplots(rows=3, cols=3, specs=[[{'type': 'scatter'}]*3]*3, subplot_titles=verifier_names)
fig1 = make_subplots(rows=3, cols=3, subplot_titles=verifier_names)
fig2 = make_subplots(rows=3, cols=3, subplot_titles=verifier_names)

for i,v in enumerate(log):
    print(v)
    res = np.zeros((5,6))
    res_time = np.zeros((5,6))
    for vpc in log[v]:
        if log[v][vpc][0] in ['sat','unsat']:
            res[int(vpc[0])][int(vpc[1])] += 1
        res_time[int(vpc[0])][int(vpc[1])] += log[v][vpc][1]
    print(res)
    print(res_time)
    xs = []
    ys = []
    zs = []
    ts = []
    for x in range(5):
        for y in range(6):
           xs += [x]
           ys += [y]
           zs += [res[x][y]]
           ts += [res_time[x][y]]

    zs = np.array(zs)
    ts = np.array(ts)
    zs = zs*10
    ts = ts/np.linalg.norm(ts)*100

    fig1.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=zs), showlegend=False),row=int(i/3)+1,col=i%3+1)
    fig2.add_trace(go.Scatter(x=xs, y=ys, mode='markers', marker=dict(size=ts), showlegend=False),row=int(i/3)+1,col=i%3+1)
    #fig = px.scatter(x=xs, y=ys, size=zs)


fig1.write_image('solutions.pdf')
fig2.write_image('time.pdf')

fig1.show()
fig2.show()
