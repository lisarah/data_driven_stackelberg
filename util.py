#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 12:18:15 2024

@author: huilih
"""
import matplotlib.pyplot as plt
import numpy as np 

def plot_traj(x_trajs,legend=None):
    if type(x_trajs) is not list:
        x_trajs = [x_trajs]
    if legend is None:
        legend = ['' for _ in x_trajs]
    plt.figure()
    
    for x_traj, l_i in zip(x_trajs, legend):
        cur_len = len(x_traj)
        plt.plot([x_traj[2*j] for j in range(int(cur_len/2))], 
                 [x_traj[2*j+1] for j in range(int(cur_len/2))], label=l_i)

    plt.legend()
    plt.show(block=False)
def recovery_norm(g, y, H_mat):
    return np.linalg.norm(H_mat.dot(g) - y)