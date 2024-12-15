#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 22:51:23 2024

@author: sadafmoaveninejad
"""

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import welch#, hanning

from scipy.signal import welch, hamming
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, explained_variance_score

#%%
# --------- Define synthetic time series generation functions (WCF, TLF, fBm, fGn)


def generate_wcf(FDth, N=1000, lamda=5, M=26, amplitude=1):
    t = np.linspace(0, 1, N)
    H = 2 - FDth
    x_wsc = np.zeros(N)
    for i in range(N):
        x_wsc[i] = np.sum((lamda ** -((np.arange(M+1) * H))) * np.cos(2 * np.pi * (lamda ** (np.arange(M+1))) * t[i]))
    
    # Normalize the signal to an amplitude of 1 and then scale
    #x_wsc = x_wsc* amplitude
    x_wsc = (x_wsc / np.max(np.abs(x_wsc))) * amplitude
    
    return x_wsc

def generate_tlf(FD, fs, x=None, step=0.005, amplitude=1):
    H = 2 - FD  # Calculate H based on the provided fractal dimension
    w = 2 ** (-H)  # Compute w
    if x is None:
        x = 1   # Set default x if not provided
    step = 1/fs  # Define step based on the sampling frequency
    n_final = 10  # Default maximum summation limit

    # Initialize arrays to store values
    x_values = np.arange(0, x + step, step)
    B = np.zeros_like(x_values)  # Initialize Blancmange values array

    # Compute Blancmange-Takagi function
    for R, x1 in enumerate(x_values):
        org = np.array([2 ** n * x1 for n in range(n_final + 1)])
        a1 = w ** np.arange(n_final + 1) * np.abs(org - np.round(org))
        B[R] = np.sum(a1)

    # Normalize and scale the signal by the amplitude
    #B = B * amplitude
    B = (B / np.max(np.abs(B))) * amplitude

    return B[:-1]  # Return the adjusted signal


from fbm import FBM

def generate_fbm(FDth, N=1000, amplitude=1):
    """
    Generate fractional Brownian motion (fBm) with specified amplitude.
    """
    hurst = 2 - FDth  # Calculate Hurst parameter from the fractal dimension
    f = FBM(n=N-1, hurst=hurst, method='daviesharte')
    fbm = f.fbm() 
    # Normalize and scale
    fbm = (fbm / np.max(np.abs(fbm))) * amplitude
    return fbm


def generate_fgn(FDth, N=1000, amplitude=1):
    """
    Generate Fractional Brownian Motion (fBm) from fractional Gaussian noise (fGn).
    """
    hurst = 2 - FDth
    f = FBM(n=N-1, hurst=hurst, method='daviesharte')
    fgn = f.fgn() * amplitude
    # Normalize and scale
    fgn = (fgn / np.max(np.abs(fgn))) * amplitude
    return fgn


#%% Methods

# ------ Define fractal dimension estimation functions (sPSD, DFA, GHE, HFD, KFD, BC)


def nextpow2(n):
    """Calculate the next power of 2 for a given number."""
    return int(np.ceil(np.log2(n)))

def compute_window_length(window_length):
    """Calculate window length to be the max of 256 or the next power of two greater than window_length."""
    return max(256, 2**nextpow2(window_length))


def sPSD(data, fs):
    # Calculate the appropriate segment length
    proposed_nperseg = max(256, 2**np.ceil(np.log2(len(data) / 8)))
    #welch_n_per_seg = int(min(proposed_nperseg, len(data)))
    welch_n_per_seg = len(data)
    # Set 50% overlap and ensure FFT length is a power of two at least as large as nperseg
    welch_n_overlap = welch_n_per_seg // 2
    welch_n_fft = max(welch_n_per_seg, 2**int(np.ceil(np.log2(proposed_nperseg))))
    #print(welch_n_per_seg,welch_n_fft )
    # Select the window, defaulting to Hamming unless the segment size is 1
    window = hamming(welch_n_per_seg) if welch_n_per_seg > 1 else np.array([1])

    # Compute the power spectral density (PSD)
    f, Pxx = welch(data, fs=fs, window=window, nperseg=welch_n_per_seg, noverlap=welch_n_overlap, nfft=welch_n_fft, scaling='density')

    # Filter out the zero frequency and ensure no zero power spectral densities
    valid_mask = (f > 0) & (Pxx > 0)
    if not np.any(valid_mask):
        print("No valid data for logarithmic transformation.")
        return None

    log_F = np.log10(f[valid_mask])
    log_Pxx = np.log10(Pxx[valid_mask])

    # Fit a line to the log-log plot
    Xpred = np.column_stack((np.ones(log_F.shape), log_F))
    linreg = LinearRegression()
    linreg.fit(Xpred, log_Pxx)
    betta = linreg.coef_[1]

    # The slope of the PSD in MATLAB's pwelch is considered negative
    slope = -betta

    return slope
 

def dfa(data, win_length, order=1):
    N = len(data)
    n = N // win_length  # Number of windows
    N1 = n * win_length  # Adjusted length of the data for the windows
    y = np.cumsum(data[:N1] - np.mean(data[:N1]))  # Cumulative sum adjusted by the mean

    # Fit a polynomial to each window of the data
    fitcoef = np.zeros((n, order + 1))
    Yn = np.zeros(N1)
    for j in range(n):
        window_slice = slice(j * win_length, (j + 1) * win_length)
        fitcoef[j, :] = np.polyfit(np.arange(1, win_length + 1), y[window_slice], order)
        Yn[window_slice] = np.polyval(fitcoef[j, :], np.arange(1, win_length + 1))

    # Calculate the root mean square fluctuation
    rms_fluctuation = np.sqrt(np.mean((y - Yn) ** 2))

    return rms_fluctuation

def DFA(DATA,fs):
    win_lengths = np.arange(10, len(DATA), 5)  # Window lengths
    F_n = np.array([dfa(DATA, int(wl), 1) for wl in win_lengths])  # Calculate DFA for each window length

    # Perform log-log linear fit
    A = np.polyfit(np.log(win_lengths), np.log(F_n), 1)
    Alpha1 = A[0]
    #D = 3 - Alpha1

    return Alpha1


def GHE(S, fs, q=1, max_T=20):
    """
    Calculate the Generalized Hurst Exponent (GHE) for a given signal and derive the Fractal Dimension (FD).

    Parameters:
    - S: array-like
        The input signal (1D array).
    - fs: int
        Sampling frequency of the signal (not used in this implementation but kept for compatibility).
    - q: int or array-like
        The moment order(s) for the Generalized Hurst Exponent (default is 1).
    - max_T: int
        Maximum scale for the fluctuation analysis (default is 19).

    Returns:
    - FD: list of dicts
        The fractal dimension(s) calculated as FD = 2 - mean H(q), formatted without brackets for values.
    """
    # Convert to numpy array
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 1:
        raise ValueError("S must be a 1D array.")

    L = len(S)
    H = np.zeros((max_T - 4, len(np.atleast_1d(q))))
    k = 0

    for Tmax in range(5, max_T + 1):
        x = np.arange(1, Tmax + 1)
        mcord = np.zeros((Tmax, len(np.atleast_1d(q))))

        for tt in range(1, Tmax + 1):
            dV = S[tt:] - S[:-tt]
            VV = S[:-tt]
            N = len(dV)
            X = np.arange(1, N + 1)
            Y = VV
            mx = np.mean(X)
            SSxx = np.sum((X - mx)**2)
            my = np.mean(Y)
            SSxy = np.sum((X - mx) * (Y - my))
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            dVd = dV - cc1
            VVVd = VV - cc1 * X - cc2
            for qq, q_val in enumerate(np.atleast_1d(q)):
                mcord[tt - 1, qq] = np.mean(np.abs(dVd)**q_val) / np.mean(np.abs(VVVd)**q_val)

        mx = np.mean(np.log10(x))
        SSxx = np.sum((np.log10(x) - mx)**2)

        for qq in range(len(np.atleast_1d(q))):
            my = np.mean(np.log10(mcord[:, qq]))
            SSxy = np.sum((np.log10(x) - mx) * (np.log10(mcord[:, qq]) - my))
            H[k, qq] = SSxy / SSxx

        k += 1

    mH = np.mean(H, axis=0) / q
    FD = 2 - mH

    # Convert FD output to a flat dictionary of formatted results without brackets
    #result = []
    for i, value in enumerate(FD):
        result=np.round(value, 2)

    return result


def HE(S, fs, max_T=19):
    """
    Calculate the Fractal Dimension (FD) using the Hurst exponent.

    Parameters:
    - S: array-like
        The input signal (1D array).
    - fs: int
        Sampling frequency of the signal (not used in this implementation but kept for compatibility).
    - max_T: int
        Maximum scale for the fluctuation analysis (default is 30).

    Returns:
    - FD: float
        The fractal dimension calculated as FD = 2 - H.
    """
    # Convert to numpy array
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 1:
        raise ValueError("S must be a 1D array.")

    L = len(S)
    H = np.zeros(max_T - 4)

    for Tmax in range(5, max_T + 1):
        x = np.arange(1, Tmax + 1)
        mcord = np.zeros(Tmax)

        for tt in range(1, Tmax + 1):
            dV = S[tt:] - S[:-tt]
            VV = S[:-tt]
            N = len(dV)
            X = np.arange(1, N + 1)
            Y = VV
            mx = np.mean(X)
            SSxx = np.sum((X - mx)**2)
            my = np.mean(Y)
            SSxy = np.sum((X - mx) * (Y - my))
            cc1 = SSxy / SSxx
            cc2 = my - cc1 * mx
            dVd = dV - cc1
            VVVd = VV - cc1 * X - cc2

            mcord[tt - 1] = np.mean(np.abs(dVd)**2) / np.mean(np.abs(VVVd)**2)

        mx = np.mean(np.log10(x))
        SSxx = np.sum((np.log10(x) - mx)**2)
        my = np.mean(np.log10(mcord))
        SSxy = np.sum((np.log10(x) - mx) * (np.log10(mcord) - my))
        H[Tmax - 5] = SSxy / SSxx

    # Format H values to two decimal places
    H = np.round(H, 2)
    mH = np.mean(H)
    FD = 2 - mH
    return FD
        

def HFD(time_series, fs, max_k=5):
    """
    Compute the Higuchi Fractal Dimension (HFD) for a given time series.
    
    :param time_series: 1D numpy array containing the time series data.
    :param max_k: Maximum value of k (segment size) to consider.
    :return: Estimated Higuchi Fractal Dimension.
    """
    def Lmk(m, k):
        """
        Calculate the length of the curve for a given segment size k and start index m.
        """
        N = len(time_series)
        num_segments_m = int(np.floor((N - m) / k))
        summation = sum(abs(time_series[m + i * k - 1] - time_series[m + (i - 1) * k - 1]) for i in range(1, num_segments_m + 1))
        Ng = (N - 1) / (num_segments_m * k)
        Lmk_value = (summation * Ng) / k
        return Lmk_value

    def mean_Lk(k):
        """
        Calculate the mean length of the curve for a given segment size k, averaging over all possible start indices.
        """
        L_values = [Lmk(m, k) for m in range(1, k + 1)]
        L_values = [l for l in L_values if l is not None]
        return np.mean(L_values) if L_values else 0

    # Calculate mean length of the curve for different segment sizes
    L_values = [mean_Lk(k) for k in range(1, max_k + 1)]
    valid_L_values = [l for l in L_values if l > 0]

    if not valid_L_values:
        print("Insufficient valid L(k) values.")
        return None

    # Perform log-log linear regression
    log_k_values = np.log10(1 / np.array(range(1, max_k + 1)))
    log_L_values = np.log10(valid_L_values)
    slope, _, _, _, _ = linregress(log_k_values, log_L_values)

    # Truncate slope to two decimal places
    return round(slope, 2)


def KFD(signal,fs):
    """ Katz's Fractal Dimension (KFD) """
    n = len(signal)
    distances = np.sqrt((np.arange(n) - 0)**2 + (signal - signal[0])**2)
    L = np.sum(np.sqrt(np.diff(signal)**2 + 1))
    a = L / (n - 1)
    d = np.max(distances)
    kfd = np.log10(n) / (np.log10(n) + np.log10(d / L))
    return kfd

def compute_fractal_dimension(method, signal,fs):
    """
    Compute the fractal dimension of a signal using the specified method.
    
    :param method: String representing the fractal dimension estimation method.
    :param signal: 1D numpy array representing the time series.
    :return: Estimated fractal dimension.
    """
    methods = {
        'sPSD': sPSD,
        'DFA': DFA,
        'HE': HE,
        'GHE': GHE,
        'HFD': HFD,
        'KFD': KFD#,
        #'BC': BC
    }
    
    if method in methods:
        return methods[method](signal,fs)
    else:
        raise ValueError(f"Method {method} is not recognized.")