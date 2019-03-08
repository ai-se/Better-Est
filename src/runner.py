from __future__ import division, print_function

import numpy as np
from pdb import set_trace
from demos import cmd
import pickle
import matplotlib.pyplot as plt
from os import listdir
import random

from collections import Counter
import time
from mar import MAR
from sk import rdivDemo
import pandas as pd

def TEST_AL(filename, old_files = [], stop='est', stopat=1, error='none', interval = 100000, starting =1, seed=0, timestart = False, step =10):
    stopat = float(stopat)
    thres = 0
    counter = 0
    pos_last = 0
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename,old_files)
    read.step = step

    read.interval = interval



    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False

    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        if pos < starting or pos+neg<thres:
            if timestart:
                for id in read.fast():
                    read.code_error(id, error=error)
            else:
                for id in read.random():
                    read.code_error(id, error=error)
        else:
            a,b,c,d =read.train(weighting=True,pne=True)
            if stop == 'est':
                if stopat * read.est_num <= pos:
                    break
            else:
                if pos >= target:
                    break
            for id in c:
                read.code_error(id, error=error)
    # read.export()
    # results = analyze(read)
    # print(results)
    # read.plot()
    return read

def Supervised(filename, old_files = [], stop='est', stopat=1, error='none', interval = 100000, starting =1, seed=0, timestart = False, step =10):
    stopat = float(stopat)
    np.random.seed(seed)

    read = MAR()
    read = read.create(filename,old_files)
    read.step = step

    read.interval = interval



    num2 = read.get_allpos()
    target = int(num2 * stopat)
    if stop == 'est':
        read.enable_est = True
    else:
        read.enable_est = False


    read.train_supervised()
    pos, neg, total = read.get_numbers()

    read.query_supervised()
    read.record['est'][0]= read.est_num


    while True:
        pos, neg, total = read.get_numbers()
        try:
            print("%d, %d, %d" %(pos,pos+neg, read.est_num))
        except:
            print("%d, %d" %(pos,pos+neg))

        if pos + neg >= total:
            break

        if stop == 'est':
            if stopat * read.est_num <= pos:
                break
        else:
            if pos >= target:
                break
        for id in read.query_supervised()[:read.step]:
            read.code_error(id, error=error)
    return read

def Plot(results,file_save):
    font = {'family': 'normal',
                'weight': 'bold',
                'size': 20}

    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': False, 'figure.figsize': (16, 8)}

    plt.rcParams.update(paras)

    fig = plt.figure()
    ax=plt.subplot(111)
    pos=results['true'][0]
    total = results['true'][1]
    colors=['red','blue','green']
    i=0
    for key in results:
        if key == 'true':
            continue

        x= np.array(map(float,results[key]['x']))/total
        y= np.array(map(float,results[key]['pos']))/pos
        ax.plot(x, y, color=colors[i],linestyle = '-', label=key)
        if len(results[key]['est'])>1:
            z= np.array(map(float,results[key]['est']))/pos
            ax.plot(x, z, color=colors[i],linestyle = ':')
        i+=1

    plt.subplots_adjust(top=0.95, left=0.12, bottom=0.2, right=0.75)
    ax.legend(bbox_to_anchor=(1.02, 1), loc=2, ncol=1, borderaxespad=0.)
    plt.ylabel("Recall")
    plt.xlabel("Cost")


    plt.savefig('../figure/'+file_save+".png")
    plt.savefig('../figure/'+file_save+".pdf")
    plt.close(fig)




### metrics

def APFD(read):
    n = len(read.body["code"][:read.newpart])
    m = Counter(read.body["code"][:read.newpart])["yes"]
    order = np.argsort(read.body['time'][:read.newpart])
    time = 0
    apfd = 0
    apfdc = 0
    num = 0
    time_total = sum(read.body['duration'][:read.newpart])
    for id in order:
        if read.body["code"][id] == 'undetermined':
            continue
        time += read.body["duration"][id]
        num+=1
        if read.body["code"][id] == 'yes':
            apfd += (num)
            apfdc += time_total - time + read.body["duration"][id] / 2

    apfd = 1-float(apfd)/n/m+1/(2*n)
    apfdc = apfdc / time_total / m
    return apfd, apfdc

def CostRecall(read, recall):
    m = Counter(read.body["code"][:read.newpart])["yes"]
    order = np.argsort(read.body['time'][:read.newpart])
    time_total = sum(read.body['duration'][:read.newpart])
    target = recall * m
    time = 0
    pos = 0
    for id in order:
        if read.body["code"][id] == 'undetermined':
            continue
        time += read.body["duration"][id]
        if read.body["code"][id] == 'yes':
            pos += 1
        if pos>=target:
            return time/time_total





def metrics(read, results, treatment, cost=0):
    results["APFD"][treatment], results["APFDc"][treatment] = APFD(read)
    results["X50"][treatment] = CostRecall(read, 0.5)
    results["FIRSTFAIL"][treatment] = CostRecall(read, 0.00000000001)
    results["ALLFAIL"][treatment] = CostRecall(read, .99999999999)
    results["runtime"][treatment] = cost
    return results




### exp

def exp_HPC(i , input = '../data/'):
    files = listdir(input)
    file = files[i]
    files.remove(file)
    results = {}

    read = Supervised(file, files)
    pos = Counter(read.body['label'][:read.newpart])['yes']
    total = read.newpart

    results['true']=[pos,total]
    results['supervised'] = read.record

    read = Supervised(file, files)
    results['active'] = read.record


    # Plot(read,file.split('.')[0])
    with open("../dump/"+'.'.join(file.split('.')[:-1])+'.csv',"w") as handle:
        pickle.dump(results,handle)



def plot_HPC(input = '../dump/'):
    files = listdir(input)
    for file in files:
        with open("../dump/"+file,"r") as handle:
            result = pickle.load(handle)
        Plot(result,'.'.join(file.split('.')[:-1]))



def collect_HPC(path = "./dump/"):
    files = listdir(path)
    results = {}
    for i, file in enumerate(files):
        with open(path+file,"r") as handle:
            one_run = pickle.load(handle)
        if i==0:
            for metrics in one_run:
                results[metrics] = {}
                for treatment in one_run[metrics]:
                    results[metrics][treatment] = [one_run[metrics][treatment]]
        else:
            for metrics in one_run:
                for treatment in one_run[metrics]:
                    results[metrics][treatment].append(one_run[metrics][treatment])

    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            try:
                summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
            except:
                set_trace()
    print("summary:")
    print(summary)
    with open("./result/result.pickle","w") as handle:
        pickle.dump(results,handle)

def sum_HPC():
    with open("./result/result.pickle","r") as handle:
        results = pickle.load(handle)
    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
    print("summary:")
    print(summary)

    for metrics in results:
        print(metrics)
        if metrics == "X50":
            rdivDemo(results[metrics],bigger_better=False)
        else:
            rdivDemo(results[metrics],bigger_better=False)

def sum_relative():
    with open("./dump/result.pickle","r") as handle:
        results = pickle.load(handle)

    result = {}
    for metrics in results:
        result[metrics] = {}
        for t in results[metrics]:
            if t=='A2':
                continue
            result[metrics][t] = np.array(results[metrics][t]) / np.array(results[metrics]['A2'])
    results = result
    summary = {}
    for metrics in results:
        summary[metrics] = {}
        for t in results[metrics]:
            summary[metrics][t]={'median': np.median(results[metrics][t]), 'iqr': np.percentile(results[metrics][t],75)-np.percentile(results[metrics][t],25)}
    print("summary:")
    print(summary)

    for metrics in results:
        print(metrics)
        if metrics == "X50":
            rdivDemo(results[metrics],bigger_better=False)
        else:
            rdivDemo(results[metrics],bigger_better=False)


def exp_target(input = '../data/',target='apache-ant-1.7.0.csv'):
    files = listdir(input)
    files.remove(target)
    try:
        files.remove('.DS_Store')
    except:
        pass
    read = TEST_AL(target)
    set_trace()
    est = read.record['est'][0]
    print(target+": "+str(est)+" / "+ str(read.record['pos'][-1])+" / "+str(read.record['pos'][-1]))
    print(str(read.record['x'][-1])+" / "+ str(read.newpart))


def exp_all(input = '../data/'):
    files = listdir(input)
    for file in files:
        exp_target(input=input,target=file)



if __name__ == "__main__":
    eval(cmd())
