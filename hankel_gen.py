#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:25:42 2024

@author: Sarah Li
"""
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as sp
from enum import Enum

class opp(Enum):
    NEUTRAL=1
    AGGRESSIVE=2
    AVOID=3

def GJ_gen(A, B, data_len):
    x_len, u_len = B.shape
    J =  np.eye(x_len)
    A_t = np.eye(x_len)
    
    G= np.zeros((x_len, u_len*data_len))
    G_row= np.zeros((x_len, u_len*data_len))
    for t in range(data_len):
        G_row = np.hstack((A_t.dot(B), G_row[:,:(data_len-1)*u_len]))
        A_t = A_t.dot(A)
        J = np.vstack((J, A_t))
        G = np.vstack((G, G_row))
        
    return G, J   

def traj_gen(G, J, Q, R, x_0, tau, T, data_len, w=0, U_leader=None, noise=0, 
              opp_type=opp.NEUTRAL):
    # tau is x_L for aggressive
    # tau is reference for neutral
    x_len, _ = Q.shape
    u_len, _ = R.shape
    Q_hat = np.kron(Q, np.eye(data_len + 1))
    R_hat = np.kron(R, np.eye(data_len))
    M = G.T.dot(Q_hat).dot(G) + R_hat
    
    if opp_type is opp.NEUTRAL:
        MU_star = G.T.dot(Q_hat).dot(tau - J.dot(x_0)) 
        U_star = np.linalg.pinv(M).dot(MU_star)
        X_star = G.dot(U_star) + J.dot(x_0)
    elif opp_type is opp.AGGRESSIVE:
        # MU_star = G.T.dot(Q_hat).dot(U_leader - J.dot(x_0) ) 
        # U_star = np.linalg.pinv(M).dot(MU_star)
        MU_star = G.T.dot(Q_hat).dot(U_leader) 
        U_star = np.linalg.pinv(M).dot(MU_star)
        X_star = G.dot(U_star)  
    elif opp_type is opp.AVOID: 
        MU_star = G.T.dot(Q_hat.dot(J.dot(x_0) - J.dot(x_0)) - w*U_leader) 
        U_star = np.linalg.pinv(M).dot(MU_star)
        X_star = G.dot(U_star) + J.dot(x_0)
    X_star = X_star[2:]
    return X_star, sp.hankel(X_star[:x_len*T], X_star[x_len*T:]) 

def tau_gen(x_0, data_len, delta):
    reference_traj = [np.array([-t/delta, t/delta] + x_0) 
                      for t in range(data_len+1)]
    tau = np.concatenate(reference_traj, axis=0)
    return tau
    
def leader_gen(data_len, y_0, u_len, T, noise=0):
    U_traj = [y_0 + np.array([-t,-t]) + noise*(np.random.rand(u_len)-0.5) 
          for t in range(data_len+1)]
    U_traj = np.concatenate(U_traj, axis=0)
    return U_traj, sp.hankel(U_traj[2:u_len*T+2], U_traj[u_len*T+2:])  

def leader_rand(data_len, y_0, x_0, u_len, T, noise=0):
    U_traj = [y_0 - x_0 + np.array([-t, -t]) for t in range(int(data_len/2))]
    U_traj2 = [U_traj[-1] + 1*(np.random.rand(u_len)-0.5*np.ones(2)) 
               for t in range(data_len+1-int(data_len/2))]
    U_traj = np.concatenate(U_traj+U_traj2, axis=0)
    return U_traj, sp.hankel(U_traj[2:u_len*T+2], U_traj[u_len*T+2:])    
    
# def aggressive_gen(data_len, x_len, u_len, T, x_0, tau=None, noise=0):
    # A = np.eye(x_len)
    # # A[0,1] = 0.2
    # B = np.eye(u_len)
    # G, J = GJ_gen(A,B, data_len)
    
#     Q = np.eye(2)
#     R = np.eye(2)
#     Q_hat = np.kron(Q, np.eye(data_len + 1))
#     R_hat = np.kron(R, np.eye(data_len))

#     M = G.T.dot(Q_hat).dot(G) + R_hat
#     U_star = np.linalg.inv(M).dot(G.T).dot(Q_hat).dot(tau - J.dot(x_0))
#     X_star = G.dot(U_star) + J.dot(x_0)
#     X_star = X_star[2:]
    
#     # rank: np.linalg.matrix_rank(H_u)
#     return sp.hankel(X_star[:x_len*T], X_star[x_len*T:])
