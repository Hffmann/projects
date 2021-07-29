import control as ctrl
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import cont2discrete, lti, dlti, dstep
import math
import cmath

def onpick(event):
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind[0]
    points = tuple(zip(xdata[ind], ydata[ind]))
    print('onpick points:', points)


def step_info(t, y, SettlingTimeThreshold, RiseTimeLimits, Amp):

  InfValue = y[-1]

  #RiseTime
  tr_lower_index = (np.where(y >= RiseTimeLimits[0] * InfValue)[0])[0]
  tr_upper_index = (np.where(y >= RiseTimeLimits[1] * InfValue)[0])[0]
  RiseTime = t[tr_upper_index] - t[tr_lower_index]

  # SettlingTime
  sup_margin = (1. + SettlingTimeThreshold) * InfValue
  inf_margin = (1. - SettlingTimeThreshold) * InfValue
  # find Steady State looking for the first point out of specified limits
  for i in reversed(range(t.size)):
      if((y[i] <= inf_margin) | (y[i] >= sup_margin)):
          SettlingTime = t[i + 1]
          break
  # Peak
  PeakIndex = np.abs(y).argmax()
  PeakValue = y[PeakIndex]
  PeakTime = t[PeakIndex]
  SettlingMax = (y).max()
  SettlingMin = (y[tr_upper_index:]).min()
  # I'm really not very confident about UnderShoot:
  UnderShoot = y.min()
  OverShoot = 100. * (y.max() - InfValue) / (InfValue - y[0])

  # Eest
  Eest = Amp-InfValue

  # Return as a dictionary
  S = {
      #'RiseTime': RiseTime,
      'Tempo de Estabilização': SettlingTime,
      #'SettlingMin': SettlingMin,
      #'SettlingMax': SettlingMax,
      'Overshoot(%)': OverShoot,
      #'Undershoot': UnderShoot,
      'Pico': PeakValue,
      'Tempo de Pico': PeakTime,
      'Valor de Estabilização': InfValue,
      'Erro Estacionário': Eest
    }

  return S

def plot_step(sys, T, te=0.02, ts = (0.1,0.9), Amp=1):
  xs, ys = ctrl.step_response(sys, T=T, T_num=100)
  fig, ax = plt.subplots(figsize=(7,3))
  ax.plot(xs, ys, linewidth=1.5)
  S = step_info(xs, ys, te, ts, Amp)

  return xs, ys, S

def step_z(sysz,T, figx=9, figy=4, te=0.02, ts = (0.1,0.9), Amp=1):
  xz, yz = ctrl.step_response(Amp*sysz, T, T_num=100)
  fig, ax = plt.subplots(figsize=(figx,figy))

  ax.step(xz, yz, where='post')
  ax.plot(xz, yz, 'C0o',alpha=0)
  S = step_info(xz, yz, te, ts, Amp)
  return S

# Parameters
La = 0.4
Ra = 1.5
Va = 220
kt = 0.5
kv = 16
JM = 0.01
JL = 0.03
BM = 0.12
BL = 0.3

J = JM + JL
B = BM + BL

fs = 125
Ts = 1/fs

# First transfer function
nums = [kt]
dens = [La*J, (La*B) + (Ra*J), (Ra*B) + (kv*kt)]
Gs_1 = ctrl.TransferFunction(nums, dens)
Gz_1 = ctrl.sample_system(Gs_1, Ts, method='zoh', prewarp_frequency=None)
print(Gs_1)
print(Gz_1)
Ts_1 = Va*Gs_1


# Second transfer function
nums = [J, B]
dens = [La*J, (La*B) + (Ra*J), (Ra*B) + (kv*kt)]
Gs_2 = ctrl.TransferFunction(nums, dens)
Gz_2 = ctrl.sample_system(Gs_2, Ts, method='zoh', prewarp_frequency=None)
print(Gs_2)
print(Gz_2)
Ts_2 = Va*Gs_2



# Compensator project
controller = "PID"
# while(True):
    # C_k = input("Please enter the new compensator gain value: ")
    # C_k = int(C_k)

if controller == "PID":
    w_num = [1, -1]
    w_den = [Ts]
    W = ctrl.TransferFunction(w_num, w_den, Ts)

    # print(W, pow(W,2))
    #
    # r_pole = input("Please enter the new compensator real pole value: ")
    # i_pole = input("Please enter the new compensator im pole value: ")
    # r_pole = float(r_pole)
    # i_pole = float(i_pole)

    C_k = 300
    r_pole = 0.88
    i_pole = 0.13

    w = (complex(r_pole, i_pole) - complex(1,0))/Ts

    print((w))
    K = ((-w*Ts)-1)/((w*(1-(pow(complex(r_pole, 0),2) + pow(complex(i_pole, 0),2)))) + ((pow(w,2))*Ts))
    print(K)
    y = cmath.sqrt(K*Ts).real
    x = ((K*(1-(pow(complex(r_pole, 0),2) + pow(complex(i_pole, 0),2)))) + Ts).real
    print(x, y)

    kd = Ts - x + ((pow(y,2))/Ts)
    kp = x - (2*(Ts))
    ki = Ts
    print(kd, kp, ki)


    C = C_k*((1 + x*W + (pow(y*W,2)))/(W*(1 + (Ts*W))))
    Tz_1 = Gz_1*C
    print(Tz_1)

    # poles, zeros = ctrl.pzmap(Gz_1)
    # print(poles, zeros)
    # ctrl.rlocus(Tz_1, xlim=[-1.2,1.2], ylim=[-1,1], plotstr=True, grid=False)
    # ax = plt.gca()
    # circle = plt.Circle((0.92986332, 0.1661691), 0.05, color='r', fill=False)
    # ax.add_patch(circle)
    # plt.show()
    # ctrl.sisotool(Tz_1, plotstr_rlocus=True, rlocus_grid=False)
    # plt.show()

    sys_closed = (Tz_1).feedback(1)
    tvect, yout = ctrl.step_response(sys_closed, T=10, T_num=100)
    S = step_info(tvect, yout, 0.02, (0.1,0.9), 1)
    tvect, yout = ctrl.step_response(sys_closed, T=0.7, T_num=100)
    print(S)
    plt.step(tvect, np.squeeze(yout), mfc='none')
    ax = plt.gca()
    ax.set_xlim([0, 0.7])
    plt.show()

# Motor_c (subsystem) ##########################################################
G_1 = ctrl.tf2io(ctrl.tf(1, [La, 0]), input='d', output='i')
G_2 = ctrl.tf2io(ctrl.tf(1, [J, 0]), input='g', output='w')

sub_1 = ctrl.tf2io(ctrl.tf(Va, 1), input='v', output='a')
sub_2 = ctrl.tf2io(ctrl.tf(Ra, 1), input='i', output='b')
sub_3 = ctrl.tf2io(ctrl.tf(kv, 1), input='w', output='c')
sub_4 = ctrl.tf2io(ctrl.tf(kt, 1), input='i', output='e')
sub_5 = ctrl.tf2io(ctrl.tf(B, 1), input='w', output='f')

sum_1 = ctrl.summing_junction(inputs=['a', '-b', '-c'], output='d')
sum_2 = ctrl.summing_junction(inputs=['e', '-f'], output='g')

motor = ctrl.interconnect((G_1, G_2, sub_1, sub_2, sub_3, sub_4, sub_5, sum_1, sum_2),
 input='v', outputs=['i', 'w'])

# Motor_d (subsystem) ########################################################
G_1 = ctrl.tf2io(ctrl.tf(1, [La, 0]).sample(Ts), input='d', output='i')
G_2 = ctrl.tf2io(ctrl.tf(1, [J, 0]).sample(Ts, method="zoh"), input='g', output='w')

sub_1 = ctrl.tf2io(ctrl.tf(1, 1).sample(Ts), input='v', output='a')
sub_2 = ctrl.tf2io(ctrl.tf(Ra, 1).sample(Ts), input='i', output='b')
sub_3 = ctrl.tf2io(ctrl.tf(kv, 1).sample(Ts), input='w', output='c')
sub_4 = ctrl.tf2io(ctrl.tf(kt, 1).sample(Ts), input='i', output='e')
sub_5 = ctrl.tf2io(ctrl.tf(B, 1).sample(Ts), input='w', output='f')

sum_1 = ctrl.summing_junction(inputs=['a', '-b', '-c'], output='d')
sum_2 = ctrl.summing_junction(inputs=['e', '-f'], output='g')

motor_d = ctrl.interconnect((G_1, G_2, sub_1, sub_2, sub_3, sub_4, sub_5, sum_1, sum_2),
 input='v', outputs=['i', 'w'])

# Compensator (subsystem) ####################################################
compensator = ctrl.tf2io(C_k*ctrl.tf([kp+ki+kd, -(kp+(2*kd)), kd], [1,-1,0], Ts), input='h', output='v')


# Pre-compensator (subsystem) ################################################
pc_1 = ctrl.tf2io(ctrl.tf(10, 1, Ts), input='k', output='l')
pc_sum = ctrl.summing_junction(inputs=['l', '-w'], output='h')

# System #####################################################################
System = ctrl.interconnect((G_1, G_2, sub_1, sub_2, sub_3, sub_4, sub_5, sum_1, sum_2, pc_1, pc_sum, compensator),
 input='k', outputs=['i', 'w', 'v'])

x_1, y_1 = ctrl.step_response(System[2,0], T=1.5)
x_2, y_2 = ctrl.step_response(System[1,0], T=1.5)
fig, axis = plt.subplots(2)

axis[0].plot(x_1, y_1, linewidth=1.5)
axis[1].plot(x_2, y_2, linewidth=1.5)
axis[0].set_xlim([0,1.15])
axis[1].set_xlim([0,1.15])

plt.show()
