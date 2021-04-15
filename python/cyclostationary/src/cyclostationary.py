# derived from  https://github.com/avian2/spectrum-sensing-methods/blob/master/sensing/utils.py
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
from numpy.lib.stride_tricks import as_strided


def scd_fam(x, Np, L, N=None):
    def sliding_window(x, w, s):
        shape = (((x.shape[0] - w) // s + 1), w)
        strides = (x.strides[0] * s, x.strides[0])
        return as_strided(x, shape, strides)

    # input channelization
    xs = sliding_window(x, Np, L)
    if N is None:
        Pe = int(np.floor(int(np.log(xs.shape[0]) / np.log(2))))
        P = 2 ** Pe
        N = L * P
    else:
        P = N // L
    # xs2 = xs[0:P, :]
    # using my own code to do the Framing; to match with MATLAB version
    NN = (P - 1) * L + Np
    xx = np.zeros(NN, dtype=np.complex_)
    for i in range(N):
        xx[i] = x[i]
    xs2 = np.zeros((P, Np), dtype=np.complex_)
    for k in range(P):
        xs2[k, :] = xx[k * L: k * L + Np]

    # windowing
    w = np.hamming(Np)
    # w /= np.sqrt(np.sum(w ** 2))  # ???
    xw = xs2 * np.tile(w, (P, 1))

    # first FFT
    XF1 = np.fft.fft(xw, axis=1)
    XF1 = np.fft.fftshift(XF1, axes=1)
    # XF1 = fft(xw)
    # XF1 = fftshift(XF1)

    # calculating complex demodulates
    f = np.arange(Np) / float(Np) - .5
    t = np.arange(P) * L

    f = np.tile(f, (P, 1))
    t = np.tile(t.reshape(P, 1), (1, Np))

    XD = XF1
    XD *= np.exp(-1j * 2 * np.pi * f * t)

    # calculating conjugate products, second FFT and the final matrix
    # Sx = np.zeros((Np, 2 * N), dtype=complex)
    Sx = np.zeros((2 * Np, 2 * N), dtype=complex)
    Mp = N // Np

    for k in range(Np):
        for l in range(Np):
            # XF2 = np.fft.fft(XD[:, k] * np.conjugate(XD[:, l]))
            # XF2 = np.fft.fftshift(XF2)
            # XF2 /= P
            XM = XD[:, k] * np.conjugate(XD[:, l])
            XF2 = np.fft.fft(XM)
            XF2 = np.fft.fftshift(XF2)
            M = abs(XF2)

            # i = (k + l) // 2
            i = k + l
            a = int(((k - l) / float(Np) + 1.) * N)
            # Sx[i, a - Mp:a + Mp] = XF2[(P // 2 - Mp):(P // 2 + Mp)]
            Sx[i, a - Mp:a + Mp] = M[(P // 2 - Mp):(P // 2 + Mp)]
    return Sx


# return alpha profile of the SCD matrix
def alphaprofile(s):
    return np.amax(np.absolute(s), axis=1)


def plotscd_fam(x, Np, L, t):
    s = scd_fam(x, Np, L)
    f = np.absolute(s)
    plt.matshow(f, cmap='hot')
    plt.suptitle(t)
    plt.colorbar()
    plt.show()


# compare with precomputed solution
if __name__ == "__main__":
    def audiotest():
        # (Np, L, x, y) = np.load('audiosample.npy', allow_pickle=True)
        x = np.load('iq.npy', allow_pickle=True)
        N = 2048
        Np = 256
        L = 64
        y = []
        print("x.shape={}, Np={}, L={}".format(x.shape, Np, L))
        # f = np.absolute(scd_fam(x, Np, L))
        f = np.absolute(scd_fam(x, Np, L, N))
        # err = np.linalg.norm(f - y)
        # passfail = 'PASS' if err == 0.0 else 'FAIL'
        # print("audiotest: {} (error={})".format(passfail, err))
        # np.savetxt("scd.txt", f, fmt="%s")
        plotscd_fam(x, 256, 64, "DeepSig")


    def bpsktest():
        # x is a bpsk + noise input 
        x = np.load('../gen/noise_bpsk.npy')[0:1024]
        plotscd_fam(x, 256, 1, "BSPK")


    def main():
        import argparse
        parser = argparse.ArgumentParser(description='scf analysis')
        parser.add_argument("-t", action="store_true", help="time execution")
        args = parser.parse_args()
        if args.t:
            import timeit
            runs = 5
            print("Timing execution over {} runs".format(runs))
            print(timeit.timeit("audiotest()", number=runs, setup="from __main__ import audiotest"))
        else:
            audiotest()
        bpsktest()


    main()
