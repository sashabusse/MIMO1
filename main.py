import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from hard_detection import hard_detection
from subtask12 import subtask12
from subtask13 import subtask13


class TaskCfg:
    def __init__(self, packet_cnt, sc_cnt, student_id):
        self.packet_cnt = packet_cnt
        self.sc_cnt = sc_cnt

        # Depending of Student ID in this task, each one has personal sub-carrier offset,
        # [ Student# * 12 + 1  ... Student# * 12 + 1 + L ]
        # REMEMBER(!)     L < K - 12* StudentID
        # ------------------------------------------------------------------------------
        self.student_id = student_id
        self.sc0 = self.student_id*12+1
        self.sc1 = self.student_id*12+1+self.sc_cnt
        self.time_offset = 2*self.student_id


t_cfg = TaskCfg(
    packet_cnt=5,       # Amount of packets for transmission (default: 32)
    sc_cnt=256,         # L Number of constellation points and allocated sub-carriers
                        # for OFDM L < K - 12*StudentID (default: 256 for BPSK)
    student_id=1
)

SNR = np.arange(-10, 21, 2)     # dB	range of SNR (default -20...8 for BPSK)
BER = np.zeros((len(SNR), t_cfg.packet_cnt))


# dimensions:
# <UE antenna> x <BS antenna> x <sub-carrier> x <time>
link_channel = io.loadmat("Data/link_chan_PATH.mat")['Link_Channel']
ue_ant_cnt, bs_ant_cnt, file_sc_count, T = link_channel.shape
path_loss = np.mean(np.abs(link_channel))
# normalization of the channel
link_channel = link_channel / (np.sqrt(2.) * path_loss)


print('Channel path_loss:\t{}  ({} dB)'.format(path_loss, 20 * np.log10(path_loss)))
# --------------------------------------------------------------------------


# ================================================================
# task 1 / item 1 : load channel from QUADRIGA or Data File
# ================================================================

fig_H, ax_H = plt.subplots(2, 1)

ax_H[0].plot(range(1, t_cfg.sc_cnt + 1), np.abs(link_channel[0, 0, 0:t_cfg.sc_cnt, 0]))
ax_H[0].plot([0, t_cfg.sc_cnt], [1, 1], 'r--')

ax_H[0].grid(True)
ax_H[0].set_xlabel('subcarrier index')
ax_H[0].set_ylabel('abs(H)')
ax_H[0].set_ylim((0, 1.2))

ax_H[1].plot(range(0, t_cfg.sc_cnt), np.rad2deg(np.angle(link_channel[0, 0, 0:t_cfg.sc_cnt, 0])))

ax_H[1].grid(True)
ax_H[1].set_xlabel('subcarrier index')
ax_H[1].set_ylabel('arg(H)')

# ================================================================
# task 1 / item 2 : create spatial matching filter and output BER(SNR)
# ================================================================

for packet_ind in range(0, t_cfg.packet_cnt):
    # [1] generation information bit sequence length L and map it in BPSK (single OFDM symbol) ###
    tx_iq = 2 * np.random.randint(0, 2, t_cfg.sc_cnt) - 1

    # test reception for each SNR case
    for q in range(0, len(SNR)):
        # !!! STUDENT TASK !!!   SUBTASK 1.1
        # Necessary to create vector p such that satisfy coherent transmission in spatial domain
        # the same time, power of the weighted vector ||p}} = 1 (limited by PA usage as 1 (100#)
        #
        # WHEN YOU HAVE IMPLEMENTED YOUR CODE AND REPLACE UNIT VECTOR
        # you will be ready to run transmission emulation in frequency domain (think,
        # WHY(?) did we select frequency domain ?
        #
        p = np.ones((bs_ant_cnt, t_cfg.sc_cnt)) / np.sqrt(bs_ant_cnt)
        # p = subtask11(Link_Channel(:,:, StudentID * 12 + StudentID ^ 2, StudentID ^ 2));

        # Also we use following expression to create personal time offset [Student# * 2]
        t = t_cfg.time_offset + packet_ind

        x = np.zeros((bs_ant_cnt, t_cfg.sc_cnt), complex)
        y0 = np.zeros((ue_ant_cnt, t_cfg.sc_cnt), complex)
        for sc_ind in range(t_cfg.sc_cnt):
            x[:, sc_ind] = p[:, sc_ind]*tx_iq[sc_ind]
            y0[:, sc_ind] = link_channel[:, :, (t_cfg.sc0 - 1 + sc_ind), t].dot(x[:, sc_ind])

        # estimate power of signal to generate noise for given SNR
        Ps = np.mean(np.var(y0, axis=0))
        Dn = Ps / 10**(SNR[q]/10)

        # adding complex gaussian noise with variation Dn and expectation mean|n0| = 1
        n0 = (np.random.normal(loc=0, scale=1, size=(ue_ant_cnt, t_cfg.sc_cnt)) +
              1j * np.random.normal(loc=0, scale=1, size=(ue_ant_cnt, t_cfg.sc_cnt))) * np.sqrt(Dn / 2)

        # then we can update y0 into y with noise
        y = y0 + n0

        # Let assume that we have perfect channel estimation H
        # Channel knowledge (perfect) on UE has dimensionality MxNxL
        H = link_channel[:, :, (t_cfg.sc0 - 1):(t_cfg.sc0 - 1 + t_cfg.sc_cnt), t]

        # If we have channel estimation H, we can use straight forward way MIMO equalization
        # as we sent single stream, then each receive antenna has to have similar result, and
        # to reduce noise impact, we just sum up all antennas assuming
        # COHERENT SUMMATION for SIGNAL, and AVERAGING for noise
        rx_iq = np.zeros(t_cfg.sc_cnt, dtype=complex)
        lamb = np.zeros((ue_ant_cnt, t_cfg.sc_cnt))
        for sc_ind in range(0, t_cfg.sc_cnt):
            rx_iq[sc_ind] = np.sum(np.linalg.pinv(H[:, :, sc_ind]).dot(y[:, sc_ind]))
            U, tmp, V = np.linalg.svd(H[:, :, sc_ind])
            lamb[:, sc_ind] = tmp

        LMBD = np.sort(np.mean(lamb, 1))[::-1] / max(np.mean(lamb, 1))
        # ###########################################################################

        # We have used H for equalization and summation. Do you think we can find
        # better way to improve SNR?
        #   a) Is it possible to apply matching filter on the receiver (the same way as transmitter) ?
        #   b) What will be different, if we would like to transmit dual stream (size(s) = [2 L] ?
        # (!) TRY TO IMPLEMENT DUAL TRANSMISSION and GET +10 points.   SUBTASK 1.2
        # Two spatial streams
        #
        subtask12(link_channel[:, :, t_cfg.student_id * 12 + t_cfg.student_id ** 2, t_cfg.student_id ** 2])
        # ###########################################################################

        # hard detection procedure for BPSK
        rx_iq = hard_detection(rx_iq, "BPSK")
        # number of detected errors:
        s_err = np.sum(np.abs(tx_iq - rx_iq) / 2)
        # bit error rate
        BER[q, packet_ind] = s_err / t_cfg.sc_cnt

        print('BER = {:.2f} \t lambda_1 = {:.2f} \t lambda_2 = {:.2f}'.format(BER[q, packet_ind], LMBD[1], LMBD[2]))

    print('Packet {}'.format(packet_ind))


fig_ber, ax_ber = plt.subplots(1, 1)
ax_ber.semilogy(SNR, np.mean(BER,1))

ax_ber.grid(True)
ax_ber.set_xlabel('SNR,dB')
ax_ber.set_ylabel('BER')
ax_ber.set_ylim([1e-4, 1])
ax_ber.legend(['Reference'])


fig_eigen_val, ax_eigen_val = plt.subplots(1, 1)
ax_eigen_val.stem(range(1, 5), LMBD[0:4])
ax_eigen_val.plot([1, 4], [0.5, 0.5],'r--')

ax_eigen_val.grid(True)
ax_eigen_val.set_xlabel('Eigenvalue index in descend order')
ax_eigen_val.set_ylabel('Eigenvalue normilized magnitude')
ax_eigen_val.set_title('Eigenvalue distribution')

plt.show()




fc = 3.5e9
# ================================================================
# task 1 / item 3 : build spatial spectrum for for subchannel with
#                   following indexes:
#                   N = all; M = all;
#                   K = StudentID * 12 + StudentID^2
#                   T = StudentID^2
# ================================================================
#rho, phi, theta = subtask13(Link_Channel[:,:,StudentID*12+StudentID^2,StudentID^2],fc,2)
#
#Power_Spectrum = abs(rho)
#mesh(phi,theta,Power_Spectrum)
#ylabel('\theta');
#xlabel('\phi');
#zlabel('Spatial power spectrum,dB');
#
#
#H1 = subtask14(Link_Channel(:,:,StudentID*12+StudentID^2,StudentID^2),1);
## continue task 1.4 here to add corresponding diagrams on 'figure 2'
#[rho, phi, theta] = subtask13(H1,fc,2);
#figure(6)
#Power_Spectrum = abs(rho);
#mesh(phi,theta,Power_Spectrum);
#ylabel('\theta');
#xlabel('\phi');
#zlabel('Spatial power spectrum,dB');


# ================================================================
# END OF TASK 1 