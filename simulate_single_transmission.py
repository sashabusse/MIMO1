import numpy as np

from hard_detection import hard_detection
from subtask12 import subtask12


def simulate_single_transmission(
    t_cfg, time, snr, 
    link_channel, tx_iq
):
    ue_ant_cnt, bs_ant_cnt, file_sc_count, T = link_channel.shape

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

    x = np.zeros((bs_ant_cnt, t_cfg.sc_cnt), complex)
    y0 = np.zeros((ue_ant_cnt, t_cfg.sc_cnt), complex)
    for sc_ind in range(t_cfg.sc_cnt):
        x[:, sc_ind] = p[:, sc_ind] * tx_iq[sc_ind]
        y0[:, sc_ind] = link_channel[:, :, (t_cfg.sc0 - 1 + sc_ind), time].dot(x[:, sc_ind])

    # estimate power of signal to generate noise for given SNR
    Ps = np.mean(np.var(y0, axis=0))
    Dn = Ps / 10 ** (snr / 10)

    # adding complex gaussian noise with variation Dn and expectation mean|n0| = 1
    n0 = (np.random.normal(loc=0, scale=1, size=(ue_ant_cnt, t_cfg.sc_cnt)) +
          1j * np.random.normal(loc=0, scale=1, size=(ue_ant_cnt, t_cfg.sc_cnt))) * np.sqrt(Dn / 2)

    # then we can update y0 into y with noise
    y = y0 + n0

    # Let assume that we have perfect channel estimation H
    # Channel knowledge (perfect) on UE has dimensionality MxNxL
    H = link_channel[:, :, (t_cfg.sc0 - 1):(t_cfg.sc0 - 1 + t_cfg.sc_cnt), time]

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

    lmbd = np.sort(np.mean(lamb, 1))[::-1] / max(np.mean(lamb, 1))
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

    rx_iq = hard_detection(rx_iq, "BPSK")

    ber = np.sum(np.abs(tx_iq - rx_iq) / 2) / t_cfg.sc_cnt
    return ber, lmbd


