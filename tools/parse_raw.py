#!/usr/bin/env python

import os
import argparse
import datetime
import pathlib
import json
import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from natsort import natsorted

TIMEOUT = 600
HEADER_RESULTS = ['answer', 'time', 'sub answers', 'sub times', 'dnnv pre time']


def _parse_args():
    parser = argparse.ArgumentParser(description='GDVB raw result parser')
    parser.add_argument('task', type=str, choices=['parse', 'plot'])
    parser.add_argument('root', type=str)
    parser.add_argument('output_path', type=str)
    parser.add_argument('-v', '--verifier', type=str, choices=['ALL', 'eran_deepzono_wb', 'neurify_wb', 'nnenum_wb'])
    args = parser.parse_args()
    return args


def parse_answer_time(lines):
    verifier_answer = None
    verifier_time = None
    verifier_sub_answers = []
    sub_property_times = []

    dnnv_pre_time_start = None
    dnnv_pre_time = None

    for l in lines:
        # handle dnnv pre_time
        if dnnv_pre_time_start is None and '2021-' in l:
            time_string = l.split(' ')[5] + ' ' + l.split(' ')[6]
            dnnv_pre_time_start = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S,%f')
        if 'INS(dnnv): checking sub property: 0' in l:
            # print(l)
            idx_end = l.index(' INS(dnnv)')
            time_string = l[idx_end-19: idx_end]
            dnnv_pre_time_end = datetime.datetime.strptime(time_string, '%d/%m/%Y %H:%M:%S')
            dnnv_pre_time = (dnnv_pre_time_end - dnnv_pre_time_start).total_seconds()

        if 'INS(dnnv)' in l:
            if 'sub property result' in l:
                sub_property_result = l.strip().split(' ')[-1]
                assert sub_property_result in ['sat', 'unsat', 'unknown']
                verifier_sub_answers += [sub_property_result]
            if 'INS(dnnv): sub property took' in l:
                sub_property_times += [float(l.strip().split(' ')[-2])]
        elif '  result: ' in l:
            if 'NeurifyError' in l:
                verifier_answer = 'error'
            else:
                verifier_answer = l.strip().split(' ')[-1]
        elif '  time: ' in l:
            verifier_time = float(l.strip().split(' ')[-1])
        elif '(resmonitor) Timeout (terminating process):' in l:
            verifier_answer = 'timeout'
            verifier_time = -1
        elif '(resmonitor) Out of Memory (terminating process):' in l:
            verifier_answer = 'memout'
            verifier_time = -1

        # TODO: instrument DNNV to handle sub property sat answer
        elif 'adv found' in l:
            verifier_sub_answers += ['sat']

    if dnnv_pre_time is None:
        for l in lines:
            if 'INS(dnnv): verification starts' in l:
                time_string = l[: 19]
                dnnv_pre_time_end = datetime.datetime.strptime(time_string, '%d/%m/%Y %H:%M:%S')
                dnnv_pre_time = (dnnv_pre_time_end - dnnv_pre_time_start).total_seconds()

    assert verifier_answer in ['sat', 'unsat', 'unknown', 'error', 'timeout', 'memout'], verifier_answer
    assert verifier_time is not None
    assert dnnv_pre_time is not None

    return verifier_answer, verifier_time, verifier_sub_answers, sub_property_times, dnnv_pre_time


def parse_eran_dz_raw(lines):
    nb_states_all = []
    nb_safe_relu_all = []
    nb_unsafe_relu_all = []
    unsafe_relu_time_all = []
    layer_time_all = []

    # pre_time_all = []
    # post_time_all = []

    # init metrics for composed property, for debug purpose
    sub_prop_finish_flag = None
    nb_states = None
    nb_safe_relu = 0
    nb_unsafe_relu = 0
    unsafe_relu_time = []
    layer_time = []

    for l in lines:
        if 'INS(eran)' in l:

            # init metrics for each sub property
            if 'INS(eran): verifier starts here.' in l:
                sub_prop_finish_flag = False
                nb_states = None
                # nb_safe_relu = 0
                # nb_unsafe_relu = 0
                unsafe_relu_time = []
                layer_time = []

            # collect subproperty metrics when verifier finishes
            elif 'INS(eran): verifier ends here.' in l:
                # sub_prop_finish_flag = True
                nb_states_all += [nb_states]
                # nb_safe_relu_all += [nb_safe_relu]
                # nb_unsafe_relu_all += [nb_unsafe_relu]
                unsafe_relu_time_all += [unsafe_relu_time]
                layer_time_all += [layer_time]

            if 'INS(eran): number of executions: ' in l:
                nb_states = int(l.strip().split(' ')[-1])

            elif 'transformed node ' in l:
                layer_time += [float(l.strip().split(' ')[-2])]

        elif 'INS(elina)' in l:
            if 'unsafe ReLU took' in l:
                idx = l.index('unsafe ReLU took')+17
                try:
                    unsafe_relu_time += [float(l[idx: idx+8])]

                except ValueError:
                    # print(f'Warning: unsafe relu time ignored with line: {l}')
                    pass

            if 'INS(elina): ReLU activations(lt0, gt0, o/w): ' in l:
                tokens = l.strip().split(': ')[-1].split(', ')
                if len(tokens) != 3:
                    pass
                else:
                    try:
                        nb_safe_relu += int(tokens[0]) + int(tokens[1])
                        nb_unsafe_relu += int(tokens[2])

                    except ValueError:
                        # print(f'Warning: safe/unsafe relu count ignored with line: \t{l}')
                        pass

    if not sub_prop_finish_flag:
        nb_states_all += [nb_states]
        nb_safe_relu_all += [nb_safe_relu]
        nb_unsafe_relu_all += [nb_unsafe_relu]
        unsafe_relu_time_all += [unsafe_relu_time]
        layer_time_all += [layer_time]

    metric_lengths = [len(nb_states_all), len(nb_safe_relu_all), len(
        nb_unsafe_relu_all), len(unsafe_relu_time_all), len(layer_time_all)]

    print(nb_unsafe_relu_all)

    # assert all([x == metric_lengths[0] for x in metric_lengths])

    return nb_states_all, nb_safe_relu_all, nb_unsafe_relu_all, unsafe_relu_time_all, layer_time_all


def parse_neurify(lines):
    nb_oneshot_proof = 0
    nb_unsafe_relus_all = []
    split_times_all = []
    pre_time_start = None

    pre_time_all = []
    post_time_all = []

    for l in lines:
        if 'INS' in l:
            # pre time
            if 'verifier starts here' in l:
                nb_unsafe_relus = None
                end_flag = False
                split_times = []
                time_string = l.split(' ')[4] + ' ' + l.split(' ')[5]
                pre_time_start = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S,%f')

            elif 'one shot proof. No split needed.' in l:
                time_string = l.split(' ')[4] + ' ' + l.split(' ')[5]
                pre_time_end = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S,%f')
                pre_time = (pre_time_end - pre_time_start).total_seconds()
                post_time_start = pre_time_start
                pre_time_start = None
                pre_time_end = None
                pre_time_all += [pre_time]
                nb_oneshot_proof += 1

            # post time
            elif 'verifier ends here' in l:
                end_flag = True
                split_times_all += [split_times]
                split_times = []
                if not nb_unsafe_relus:
                    nb_unsafe_relus_all += [0]
            #    time_string = l.split(' ')[4] + ' ' + l.split(' ')[5]
            #    post_time_end = datetime.datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S,%f')
            #    post_time = (post_time_end - post_time_start).total_seconds()
            #    post_time_all += [post_time]
            #    post_time_start = None
            #    post_time_end = None

            elif 'split took' in l:
                try:
                    split_times += [float(l.strip().split(' ')[-2])]
                except ValueError:
                    # print('Warning: safe/unsafe relu count ignored with line:')
                    # print(f'\t{l}')
                    pass
            else:
                pass

        elif 'total wrong nodes' in l:
            nb_unsafe_relus = int(l.strip().split(' ')[-1])
            nb_unsafe_relus_all += [nb_unsafe_relus]

    if not end_flag:
        split_times_all += [split_times]
        split_times = []
        if not nb_unsafe_relus:
            nb_unsafe_relus_all += [0]

    nb_states = []
    nb_states += [len(x) for x in split_times_all]

    return nb_states, nb_unsafe_relus_all, pre_time_all, split_times_all


def parse_nnenum(lines):
    nb_oneshot_proof = 0
    advance_star_time = []
    for l in lines:
        if 'INS' in l:
            if 'advance star took' in l:
                idx_s = l.index('took')+5
                idx_e = l.index('seconds')-1
                advance_star_time += [float(l[idx_s:idx_e])]

        elif 'Proven safe before enumerate' in l:
            nb_oneshot_proof += 1

    if not advance_star_time:
        nb_states = 'N/A'
        average_advance_star_time = 'N/A'
    else:
        nb_states = len(advance_star_time)
        average_advance_star_time = np.mean(np.array(advance_star_time))

    return nb_states, average_advance_star_time


def parse(root, verifier):

    file_paths = natsorted(os.listdir(root))
    file_paths = [x for x in file_paths if '.out' in x]
    file_paths = [x for x in file_paths if f'T={TIMEOUT}' in x]
    if verifier != 'ALL':
        file_paths = [x for x in file_paths if verifier == x.split(':')[1].split('.')[0]]
    raw_results = []
    for file_path in file_paths:
        print(file_path)
        verifier = file_path.strip().split(':')[1].split('.')[0]
        if verifier != 'ALL':
            if verifier != verifier:
                continue
            else:
                pass

        tokens = file_path.strip().split(':')[0].split('_')

        factors = []
        levels = []

        for tk in tokens:
            # fix GDVB with bug not adding scaling factor of 'SF=' for networks with 100% neuron factor
            if tk == '1.000':
                continue
            factor = tk.split('=')[0]
            level = tk.split('=')[1]
            if factor in ['neu', 'fc', 'eps', 'prop']:
                factors += [factor]
                levels += [level]

            # ignore non GDVB factors
            elif factor in ['SF', 'T', 'M']:
                pass
            else:
                assert False, f'unknown factor {factor}'

        lines = open(os.path.join(root, file_path), 'r').readlines()
        lines2 = open(os.path.join(root, file_path[: -3]+'err'), 'r').readlines()
        lines = lines2[: 10] + lines + lines2[-10:]

        answer_time = parse_answer_time(lines)
        if verifier == 'eran_deepzono_wb':
            res = parse_eran_dz_raw(lines)
            header = HEADER_RESULTS + ['# states', '# safe relu', '# unsafe relu', 'avg. relu time', 'layer times']
        elif verifier == 'neurify_wb':
            res = parse_neurify(lines)
            header = HEADER_RESULTS + ['# states', '# unsafe relu', 'pre times', 'split times']
        elif verifier == 'nnenum_wb':
            res = parse_nnenum(lines)
            res = list(res)
            res.insert(1, 'N/A')
            res.insert(1, 'N/A')

        # res=nb_states, nb_safe_relu, nb_unsafe_relu, average_unsafe_relu_time, layer_time_all
        parsed_info = [verifier] + levels + list(answer_time) + list(res)
        # add header
        if len(raw_results) == 0:
            header = '; '.join(['verifier'] + factors + header)
            raw_results += [header+'\n']

        parsed_info_line = '; '.join([str(x) for x in parsed_info])
        raw_results += [parsed_info_line+'\n']

    return raw_results


def plot_heatmap(df):
    X = list(df['neu'])
    Y = list(df['fc'])
    Z = list(df['prop'])
    print(list(df['# unsafe relu']))
    df['# unsafe relu sub1'] = [json.loads(x)[0] for x in df['# unsafe relu']]
    df['# unsafe relu sub2'] = [np.sum(json.loads(x)) for x in df['# unsafe relu']]
    print(list(df['# unsafe relu sub1']))
    print(list(df['# unsafe relu sub2']))

    print(set(X), set(Y), set(Z))

    data = np.array(df['# unsafe relu sub1'])
    # data = data.reshape([len(set(X)), len(set(Y)), len(set(Z))])
    data = np.mean(data.reshape(-1, 5), axis=1).reshape(len(set(X)), len(set(Y))).transpose()

    fig, ax = plt.subplots(1, 1, figsize=(7, 9))
    sns.heatmap(data, square=True, linewidth=0.1, linecolor='white', ax=ax, cbar_kws={"shrink": 0.6, 'pad': 0.01})

    ax.invert_yaxis()
    xticks = [f'{x:.4f}' for x in sorted(list(set(X)))]
    yticks = [f'{x:.4f}' for x in sorted(list(set(Y)))]
    plt.xticks(ticks=np.arange(len(set(X)))+0.5, labels=xticks)
    plt.yticks(ticks=np.arange(len(set(Y)))+0.5, labels=yticks, rotation=0)
    plt.xlabel('Neuron Level Scale')
    plt.ylabel('Layer Level Scale')
    plt.title('# Unsafe ReLUs', size=16)
    plt.savefig(f'eran_deepzono_heatmap.png', bbox_inches='tight')


def plot_eran(output_path):
    lines = open(output_path, 'r').readlines()

    labels = []
    dnnv_pre_times = []
    layer_times = []
    nb_safe_relu = []
    nb_unsafe_relu = []
    levels = []
    import pandas as pd
    df = pd.read_csv(output_path, sep=';')
    df.columns = df.columns.str.strip()

    # select for s3 case study
    df = df[df['neu'].isin([2.8788, 3.0606, 3.2424, 3.4242, 3.6061, 3.7879, 3.9697])]
    df = df[df['fc'].isin([6.0909, 7.5303, 8.9697, 10.4091, 11.8485, 13.2879, 14.7273, 16.1667, 17.6061])]
    print(df)

    # plot heatmap
    plot_heatmap(df)

    # plot stacked bar
    for l in lines[1:]:
        tokens = [x.strip() for x in l.strip().split(';')]
        # print(tokens[:-2])
        levels = [()]
        labels += [f'{tokens[1]}/{tokens[2]}/{tokens[3]}']
        dnnv_pre_times += [float(tokens[8])]
        layer_time = json.loads(tokens[-1])
        layer_times += [layer_time[0]]

    plot_eran_stackedbar(dnnv_pre_times, layer_times)


def plot_eran_stackedbar(dnnv_pre_times, layer_times):
    width = 0.5
    fig, ax = plt.subplots()

    ax.bar(range(len(dnnv_pre_times)), dnnv_pre_times, width, color='green', label='dnnv_pretime')
    max_layers = np.max([len(x) for x in layer_times])

    bottom = np.array(dnnv_pre_times)
    for i in range(max_layers):
        if i > 0:
            bottom += np.array(Ys)
        Ys = []
        Cs = []
        for j, x in enumerate(layer_times):
            if not x:
                Ys += [TIMEOUT]
                bottom[j] = 0
                Cs += ['red']
            else:
                if len(x) < i + 1:
                    Ys += [0]
                else:
                    Ys += [x[i]]

                if i % 2 == 0:
                    label = 'affine'
                    Cs += ['blue']
                else:
                    label = 'relu'
                    Cs += ['orange']

        ax.bar(range(len(dnnv_pre_times)), Ys, width, color=Cs,
               alpha=(i+1)/(max_layers), label=label, bottom=bottom)

    # labels = np.array(labels)

    # plt.xticks(np.array(range(0, 500, 50)), labels[range(0, 500, 50)])
    # ax.set_xticklabels(ax.get_xticks(), rotation=-90)
    # plt.gca().axes.get_xaxis().set_visible(False)
    ax.set_ylabel('verification time')
    # ax.set_title('Scores by group and gender')
    # ax.legend()
    plt.savefig(f'eran_deepzono.png', bbox_inches='tight')

    for x in layer_times:
        print(len(x))
    exit()
    labels = ['G1', 'G2', 'G3', 'G4', 'G5']
    men_means = [20, 35, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]
    men_std = [2, 3, 4, 1, 2]
    women_std = [3, 5, 2, 3, 3]
    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots()

    ax.bar(labels, men_means, width, yerr=men_std, label='Men')
    ax.bar(labels, women_means, width, yerr=women_std, bottom=men_means,
           label='Women')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.legend()

    plt.show()


def plot_neurify(output_path):
    lines = open(output_path, 'r').readlines()

    labels = []

    import pandas as pd
    df = pd.read_csv(output_path, sep=';')
    df.columns = df.columns.str.strip()

    df = df[df['prop'].isin(range(5))]
    # select for s3 case study
    print(df)

    # plot heatmap
    plot_heatmap(df)

    # plot_neurify_stackedbar(dnnv_pre_times, layer_times)


def plot_neurify_stackedbar(output_path):
    lines = open(output_path, 'r').readlines()
    if 'eps' in output_path:
        base_idx = 9
    else:
        base_idx = 8
    print('nb tests :', len(lines)-1)

    labels = []
    dnnv_pre_time_all = []
    v_pre_time_all = []
    v_main_time_all = []
    main_color_all = []

    for l in lines[1:]:
        tokens = l.strip().split(';')
        print(tokens)
        labels += [f'{tokens[1]}/{tokens[2]}/{tokens[3]}']
        dnnv_pre_time_all += [float(tokens[base_idx])]

        v_pre_time9 = json.loads(tokens[base_idx+3])
        if len(v_pre_time9) > 0:
            v_pre_time_all += [v_pre_time9[0]]
        else:
            v_pre_time_all += [0]

        v_main_time_9 = json.loads(tokens[base_idx+4])
        if len(v_main_time_9) > 0:
            v_main_time_all += [np.sum(v_main_time_9[0])]
        else:
            v_main_time_all += [0]

        if tokens[base_idx-2] == ' []':
            main_color_all += ['blue']
        else:
            sub_prop_answers = [x[1:-1] for x in tokens[base_idx-2][2:-1].replace(' ', '').split(',')]
            if sub_prop_answers[0] == 'unsat':
                main_color_all += ['deepskyblue']
            elif sub_prop_answers[0] == 'sat':
                main_color_all += ['darkblue']
            else:
                assert False

    dnnv_pre_time_all = np.array(dnnv_pre_time_all)
    v_pre_time_all = np.array(v_pre_time_all)
    v_main_time_all = np.array(v_main_time_all)

    width = 0.5
    fig, ax = plt.subplots()
    ax.bar(range(len(dnnv_pre_time_all)), dnnv_pre_time_all, width, color='red', label='dnnv_pretime')
    ax.bar(range(len(v_pre_time_all)), v_pre_time_all, width,
           color='yellow', label='neurify_pretime', bottom=dnnv_pre_time_all)
    ax.bar(range(len(v_main_time_all)), v_main_time_all, width, color=main_color_all,
           label='neurify_main_time', bottom=dnnv_pre_time_all+v_pre_time_all)

    labels = np.array(labels)
    # plt.ylim(1000)
    # plt.xticks(np.array(range(0, 500, 50)), labels[range(0, 500, 50)])
    # ax.set_xticklabels(ax.get_xticks(), rotation=-90)
    # plt.gca().axes.get_xaxis().set_visible(False)
    ax.set_ylabel('verification time')
    # ax.set_title('Scores by group and gender')
    ax.legend()
    plt.show()
    exit()


def main(args):
    pathlib.Path(args.output_path).mkdir(exist_ok=True, parents=True)
    output_path = os.path.join(args.output_path, args.verifier+".csv")

    if args.task == 'parse':
        raw_results = parse(args.root, args.verifier)
        open(output_path, 'w').writelines(raw_results)
    elif args.task == 'plot':
        if args.verifier == 'eran_deepzono_wb':
            plot_eran(output_path)
        elif args.verifier == 'neurify_wb':
            plot_neurify(output_path)
        else:
            raise NotImplementedError
    else:
        assert False


if __name__ == '__main__':
    main(_parse_args())
