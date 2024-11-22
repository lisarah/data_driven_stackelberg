# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as sp


def plot_traj(x_trajs):
    if type(x_trajs) is not list:
        x_trajs = [x_trajs]
    plt.figure()
    for x_traj in x_trajs:
        cur_len = len(x_traj)
        plt.plot([x_traj[2*j] for j in range(int(cur_len/2))], 
                 [x_traj[2*j+1] for j in range(int(cur_len/2))])
    plt.show(block=False)
    
delta = 1 # time discretization
T_fut = 10
T_prev = 10 # how much history is collected for prediction
T = T_fut+T_prev
x_len = 2
u_len = 2
# going from (0,0) to (-10, 10) evenly
data_len = 2*T
reference_traj = [np.array([-t/delta +20, t/delta-20]) for t in range(data_len+1)]
x_0 = np.array([20, -20])
tau = np.concatenate(reference_traj, axis=0)
A = np.eye(x_len)
# A[0,1] = 0.2
B = np.eye(u_len)
J =  np.eye(x_len)
A_t = np.eye(x_len)


G= np.zeros((x_len, u_len*data_len))
G_row= np.zeros((x_len, u_len*data_len))
for t in range(data_len):
    G_row = np.hstack((A_t.dot(B), G_row[:,:(data_len-1)*u_len]))
    A_t = A_t.dot(A)
    J = np.vstack((J, A_t))
    G = np.vstack((G, G_row))


Q = np.eye(2)
R = np.eye(2)
Q_hat = np.kron(Q, np.eye(data_len + 1))
R_hat = np.kron(Q, np.eye(data_len))
noise = 0.1
M = G.T.dot(Q_hat).dot(G) + R_hat
U_star = np.linalg.inv(M).dot(G.T).dot(Q_hat).dot(tau - J.dot(x_0))
X_star = G.dot(U_star) + J.dot(x_0)
X_star = X_star[2:]
traj_len = len(X_star)
H_y = sp.hankel(X_star[:x_len*T], X_star[x_len*T:])

U_traj = np.random.rand(u_len*data_len)
H_u = sp.hankel(U_traj[:u_len*T], U_traj[u_len*T:])
# rank: np.linalg.matrix_rank(H_u)

H_up = H_u[:T_prev*u_len, :]
H_uf = H_u[T_prev*u_len:, :]
H_yp = H_y[:T_prev*x_len, :]
H_yf = H_y[T_prev*x_len:, :]

# testing different obesrvations
y_obs = np.array([x for x in X_star[:x_len*T_prev]]) + noise*(np.random.rand(x_len*T_prev) - 0.5)
y_obs_aggressive = [x_0 + np.array([-t, 2*t]) for t in range(T_prev)]
y_obs_ag = np.concatenate(y_obs_aggressive, axis=0)
u_obs = U_traj[:u_len*T_prev] + noise*(np.random.rand(x_len*T_prev) - 0.5)
# generate g
H_prev = np.vstack((H_up, H_yp))
g = np.linalg.inv(H_prev).dot(np.hstack((u_obs, y_obs)))
g_ag = np.linalg.inv(H_prev).dot(np.hstack((u_obs, y_obs_ag)))
y_pred = H_yf.dot(g)
u_pred = H_uf.dot(g)
y_pred_ag = H_yf.dot(g_ag)

plot_traj([X_star,y_obs, y_pred])
plot_traj([X_star, y_obs_ag, y_pred_ag])
def recovery_norm(g, y, H_mat):
    return np.linalg.norm(H_mat.dot(g) - y)

print(f'aggressive history recovery: {recovery_norm( g_ag, y_obs_ag, H_yp)}')
print(f'regular history recovery: {recovery_norm( g, y_obs, H_yp)}')
    
