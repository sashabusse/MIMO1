import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from concurrent import futures

from simulate_single_transmission import simulate_single_transmission
from TaskCfg import TaskCfg
from subtask13 import subtask13


# ================================================================
# task 1 / item 1 : load channel from QUADRIGA or Data File
# ================================================================

# dimensions:
# <UE antenna> x <BS antenna> x <sub-carrier> x <time>
link_channel = io.loadmat("Data/link_chan_PATH.mat")['Link_Channel']
ue_ant_cnt, bs_ant_cnt, file_sc_count, T = link_channel.shape
path_loss = np.mean(np.abs(link_channel))
# normalization of the channel
link_channel = link_channel / (np.sqrt(2.) * path_loss)

print('Channel path_loss:\t{}  ({} dB)'.format(path_loss, 20 * np.log10(path_loss)))
# --------------------------------------------------------------------------


# config parameters
t_cfg = TaskCfg(
    packet_cnt=5,  # Amount of packets for transmission (default: 32)
    sc_cnt=256,  # L Number of constellation points and allocated sub-carriers
    # for OFDM L < K - 12*StudentID (default: 256 for BPSK)
    student_id=1
)

snr_list = np.arange(-10, 21, 2)  # dB	range of SNR (default -20...8 for BPSK)
ber = np.zeros((len(snr_list), t_cfg.packet_cnt))
lmbd = np.zeros((t_cfg.packet_cnt, 4))

# ================================================================
# task 1 / item 2 : create spatial matching filter and output BER(SNR)
# ================================================================

for packet_ind in range(0, t_cfg.packet_cnt):
    print('Packet {}'.format(packet_ind))

    # [1] generation information bit sequence length L and map it in BPSK (single OFDM symbol) ###
    tx_iq = 2 * np.random.randint(0, 2, t_cfg.sc_cnt) - 1

    # test reception for each SNR case
    for snr_ind in range(len(snr_list)):

        ber[snr_ind, packet_ind], lmbd[packet_ind] = simulate_single_transmission(
            t_cfg, t_cfg.time_offset + packet_ind, snr_list[snr_ind],
            link_channel, tx_iq
        )

        print(
            'BER = {:.2f} \t lambda_1 = {:.2f} \t lambda_2 = {:.2f}'.format(
                ber[snr_ind, packet_ind], lmbd[packet_ind][1], lmbd[packet_ind][2])
        )

fig_ber, ax_ber = plt.subplots(1, 1)
ax_ber.semilogy(snr_list, np.mean(ber, 1))

ax_ber.grid(True)
ax_ber.set_xlabel('SNR,dB')
ax_ber.set_ylabel('BER')
ax_ber.set_ylim([1e-4, 1])
ax_ber.legend(['Reference'])

fig_eigen_val, ax_eigen_val = plt.subplots(1, 1)
ax_eigen_val.stem(range(1, 5), np.mean(lmbd, axis=0))
ax_eigen_val.plot([1, 4], [0.5, 0.5], 'r--')

ax_eigen_val.grid(True)
ax_eigen_val.set_xlabel('Eigenvalue index in descend order')
ax_eigen_val.set_ylabel('Eigenvalue normalized magnitude')
ax_eigen_val.set_title('Eigenvalue distribution')


fig_H, ax_H = plt.subplots(2, 1)

ax_H[0].plot(
    np.arange(1, t_cfg.sc_cnt + 1),
    np.abs(link_channel[0, 0, t_cfg.sc0:t_cfg.sc0 + t_cfg.sc_cnt, t_cfg.time_offset])
)
ax_H[0].plot([0, t_cfg.sc_cnt], [1, 1], 'r--')

ax_H[0].grid(True)
ax_H[0].set_xlabel('subcarrier index')
ax_H[0].set_ylabel('abs(H)')
ax_H[0].set_ylim((0, 1.2))

ax_H[1].plot(
    np.arange(0, t_cfg.sc_cnt),
    np.rad2deg(np.angle(link_channel[0, 0, t_cfg.sc0:t_cfg.sc0 + t_cfg.sc_cnt, t_cfg.time_offset]))
)

ax_H[1].grid(True)
ax_H[1].set_xlabel('subcarrier index')
ax_H[1].set_ylabel('arg(H)')

plt.show()

fc = 3.5e9
# ================================================================
# task 1 / item 3 : build spatial spectrum for for subchannel with
#                   following indexes:
#                   N = all; M = all;
#                   K = StudentID * 12 + StudentID^2
#                   T = StudentID^2
# ================================================================
# rho, phi, theta = subtask13(Link_Channel[:,:,StudentID*12+StudentID^2,StudentID^2],fc,2)
#
# Power_Spectrum = abs(rho)
# mesh(phi,theta,Power_Spectrum)
# ylabel('\theta');
# xlabel('\phi');
# zlabel('Spatial power spectrum,dB');
#
#
# H1 = subtask14(Link_Channel(:,:,StudentID*12+StudentID^2,StudentID^2),1);
## continue task 1.4 here to add corresponding diagrams on 'figure 2'
# [rho, phi, theta] = subtask13(H1,fc,2);
# figure(6)
# Power_Spectrum = abs(rho);
# mesh(phi,theta,Power_Spectrum);
# ylabel('\theta');
# xlabel('\phi');
# zlabel('Spatial power spectrum,dB');


# ================================================================
# END OF TASK 1
