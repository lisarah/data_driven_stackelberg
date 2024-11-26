# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import hankel_gen as hg

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
    
delta = 1 # time discretization
T_fut = 10
T_prev = 10 # how much history is collected for prediction

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
tau_hat = hg.zero_dynamics(tau, x_0, is_flat=True)

# leader trajectory is going from (10, 10) to (-10, -10)
y_0 = np.array([path_len, path_len])



# follower trajectories 1) neutral, 2) aggressive, 3) adversarial
Q = np.eye(2)
R = 8*np.eye(2)
# -------------- neutral follower ---------------------- #
# X_neutral/H_neutral are ZEROed
X_neutral, H_neutral = hg.traj_gen(G, J, Q, R, x_0, tau_hat, T, data_len, 
                                   opp_type=hg.opp.NEUTRAL)
U_leader, H_u = hg.leader_gen(data_len, y_0, x_0, u_len, T, noise=0)


# -------------- aggressive follower ---------------------- #
# U_rand and H_urand is centered around x_0
U_rand, H_urand = hg.leader_rand(data_len, y_0, x_0, u_len, T, noise=1)
# X_agg/H_agg are centered around x_0 - follower's trajectory 
X_agg, H_agg = hg.traj_gen(G, J, Q, R, x_0, tau_hat, T, data_len, 
                           U_leader=U_rand, opp_type=hg.opp.AGGRESSIVE)
# store data trajectories
Xs = [X_neutral, X_agg] # all zeroed
Us = [U_leader, U_rand] # zeroed
H_ys = [H_neutral, H_agg] # zeroed
H_us = [H_u, H_urand] # zeroed

obs_noise=0
T_obs = 10
x_len_obs = T_obs*x_len
u_len_obs = T_obs*u_len
# y is historical follower behavior
y_neutral_hat = np.array([x for x in X_neutral[:x_len_obs]])
# y_agg is observed follower behavior, is centered around x_0
y_agg_hat = np.array([x for x in X_agg[:x_len_obs]]) 
observed_follower = [# not seeing leader
       y_neutral_hat + obs_noise*(np.random.rand(x_len_obs) - 0.5),
       y_agg_hat + obs_noise*(np.random.rand(x_len_obs) - 0.5)]
       # # chasing leader
       # y_obs_ag, # + noise*(np.random.rand(x_len*T_prev) - 0.5),
       # # avoiding leader
       # y_obs_av] # + noise*(np.random.rand(x_len*T_prev) - 0.5)]

H_invs = []
H_futs = []
observed_leader = []
future_leader = []

for H, H_u, U_l in zip(H_ys, H_us, Us):  
    H_inv, H_fut = hg.deep_c(H, H_u, x_len_obs, u_len_obs)
    H_invs.append(H_inv)
    H_futs.append(H_fut)
    # observed leader is not centered around x_0
    observed_leader.append(U_l[:u_len_obs])
    future_leader.append(U_l[u_len_obs:2*u_len_obs])
# generate g
gs = []
follower_prediction = []    
for i in [0,1]:
    l_future = future_leader[i]
    f_t_hat = observed_follower[i] # centered around x_0
    f_t = [f_t_hat[i] + x_0[i%2] for i in range(len(f_t_hat))]
    l_t_hat = observed_leader[i] # centered around x_0
    l_t = [l_t_hat[i] + x_0[i%2] for i in range(len(l_t_hat))] # not zeroed
    historical_follower = Xs[i] # centered around x_0
    g = H_invs[i].dot(np.hstack((l_t_hat, f_t_hat,l_future)))
    pred_f = H_futs[i].dot(g)
    follower_prediction.append(pred_f) # zeroed
    gs.append(g)
    legend = ['follower data', 'follower observation', 'follower prediction', 
              'leader observation', 'leader future']
    plot_traj([historical_follower, f_t_hat, follower_prediction[-1], 
               l_t_hat, l_future],legend)
    print(f'recovery: {recovery_norm(gs[-1], follower_prediction[-1], H_fut)}')
        

net_Hy = np.hstack(H_ys)   
net_Hu = np.hstack(H_us)
H_inv, H_fut = hg.deep_c(net_Hy, net_Hu, x_len_obs, u_len_obs)

for s_ind in [0,1]:

    g = H_inv.dot(np.hstack((observed_leader[s_ind], observed_follower[s_ind],
                              future_leader[s_ind])))
    pred_f = H_fut.dot(g)
    legend = ['follower data', 'follower observation', 'follower prediction', 
              'leader observation', 'leader future']
    plot_traj([Xs[s_ind], observed_follower[s_ind], pred_f, 
                observed_leader[s_ind], future_leader[s_ind]],legend)

