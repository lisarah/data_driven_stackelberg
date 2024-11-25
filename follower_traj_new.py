# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import hankel_gen as hg

def plot_traj(x_trajs):
    if type(x_trajs) is not list:
        x_trajs = [x_trajs]
    plt.figure()
    for x_traj in x_trajs:
        cur_len = len(x_traj)
        plt.plot([x_traj[2*j] for j in range(int(cur_len/2))], 
                 [x_traj[2*j+1] for j in range(int(cur_len/2))])
    plt.show(block=False)
def recovery_norm(g, y, H_mat):
    return np.linalg.norm(H_mat.dot(g) - y)
    
delta = 1 # time discretization
T_fut = 10
T_prev = 10 # how much history is collected for prediction
obs_noise=0
x_len = 2
u_len = 2
T = T_fut+T_prev
data_len = 2*T
# system dynamics definition
A = np.eye(x_len)
# A[0,1] = 0.2
B = np.eye(u_len)
G, J = hg.GJ_gen(A, B, data_len)

# reference trajectory is going from (10,-10) to (-10, 10) evenly
path_len = 20
x_0 = np.array([path_len, -path_len])
tau = hg.tau_gen(x_0, data_len, delta)

# leader trajectory is going from (10, 10) to (-10, -10)
y_0 = np.array([path_len, path_len])
U_leader, H_u = hg.leader_gen(data_len, y_0, u_len, T, noise=0)


# follower trajectories 1) neutral, 2) aggressive, 3) adversarial
Q = np.eye(2)
R = np.eye(2)
X_neutral, H_neutral = hg.traj_gen(G, J, Q, R, x_0, tau, T, data_len, 
                                   opp_type=hg.opp.NEUTRAL)
U_rand, H_urand = hg.leader_rand(data_len, y_0, u_len, T, noise=0)
X_agg, H_agg = hg.traj_gen(G, J, Q, R, x_0, tau, T, data_len, 
                           U_leader=U_rand, opp_type=hg.opp.AGGRESSIVE)
si = 0
for x in X_agg:
    x += -x_0[si%2]
    si +=1

# store data trajectories
Xs = [X_neutral, X_agg] #
Us = [U_leader, U_rand]
H_ys = [H_neutral, H_agg] #
H_us = [H_u, H_urand]

y_neutral = np.array([x for x in X_neutral[:x_len*T_prev]])
y_agg = np.array([x for x in X_agg[:x_len*T_prev]])
obs = [# not seeing leader
        # y_neutral + obs_noise*(np.random.rand(x_len*T_prev) - 0.5),
       y_agg + 0*(np.random.rand(x_len*T_prev) - 0.5)]
       # # chasing leader
       # y_obs_ag, # + noise*(np.random.rand(x_len*T_prev) - 0.5),
       # # avoiding leader
       # y_obs_av] # + noise*(np.random.rand(x_len*T_prev) - 0.5)]

H_prevs = []
H_futs = []
u_hist = []
for H, H_u, U_l in zip(H_ys, H_us, Us):    
    H_yp = H[:T_prev*x_len, :]
    H_yf = H[T_prev*x_len:, :]
    H_up = H_u[:T_prev*u_len, :]
    H_uf = H_u[T_prev*u_len:, :]
    H_prevs.append(np.vstack((H_up, H_yp)))
    H_futs.append(H_yf)
    u_hist.append(U_l[:u_len*T_prev])
    
for i in [0,1]:
    # generate g
    gs = []
    y_preds = []
    for H_prev, H_fut, ob, u_obs in zip(H_prevs, H_futs, obs, u_hist):
        gs.append(np.linalg.pinv(H_prev).dot(np.hstack((u_obs, ob))))
        y_preds.append(H_fut.dot(gs[-1]))
        plot_traj([Xs[i], ob, y_preds[-1], u_obs])
        print(f'recovery: {recovery_norm(gs[-1], y_preds[-1], H_fut)}')
        
    
# net_Hy = np.hstack(H_ys)   
# net_Hu = np.hstack((H_u, H_u, H_u))
# H_up = net_Hu[:T_prev*u_len, :]
# H_uf = net_Hu[T_prev*u_len:, :]

# H_yp = net_Hy[:T_prev*x_len, :]
# H_yf = net_Hy[T_prev*x_len:, :]
# H_prev = np.vstack((H_up, H_yp))
# g = np.linalg.pinv(H_prev).dot(np.hstack((u_obs, obs[0])))
# y_pred = H_yf.dot(g)
# plot_traj([X_star, obs[0], y_pred])
# plot_traj([X_star, y_obs_ag, y_pred_ag])
# plot_traj([X_star, y_obs_av, y_pred_av])

# print(f'aggressive history recovery: {recovery_norm( g_ag, y_obs_ag, H_yp)}')
# print(f'regular history recovery: {recovery_norm( g, y_obs, H_yp)}')
# print(f'avoid history recovery: {recovery_norm( g_av, y_obs_av, H_yp)}')
    
