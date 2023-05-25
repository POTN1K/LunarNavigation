import numpy as np

# This code calculates the SNR margin of a link budget between the satellite and the user. It is very basic and basically a big calculator.
# By Jasper Geijsbers

# my values
# Ptr_dB = 15 #dBW  15dBW from Characterization of On-Orbit GPS and 26.8 dBW !!! this inclused the Gt. (GPSReceiverArchitecturesandMeasurements)
# Ll = 0.5 #assumed
# Lr = 0.5 #assumed
# BOLTZMANN = 1.38065E-23
# f = 2500e6 #from specification of lunar architecture
# c = 299792458
# lamba = c/f
# D_tr = 0.098
# # D_tr = 9
# L_tr = 0.212
# Gr_dB = 0.2 #10 degree pointing offset angle not clear (A Dual-Band Rectangular CPW Folded Slot Antenna for GNSS Applications)
# eta = 0.7  #helical (slides)
# et_tr = 13 #degrees assumed (not clear)
# et_re = 50 #degrees assumed (not clear)
# Tsys = 175.84 #W named Ts is "Characterization of On-Orbit GPS"
# Rm_km = 1737.500 #km
# d_lagrange_km = np.array([61000, 65676.201, 22762.600, 24438.347, 385001], dtype='int64')
# R = 50 #bit/s, see GPSReceiverArchitecturesandMeasurements.


# slides values
Ptr = 2
Ll = 0.8 
Lr = 0.8 
BOLTZMANN = 1.38065E-23
f = 2500e6 #from specification of lunar architecture
c = 3e8
lamba = c/f
D_tr = 0.5
D_r=10
# D_tr = 9
# L_tr = 0.212
eta = 0.55 #parabolic
et_tr = 0.25 #degrees assumed (not clear)
et_re = 50 #degrees assumed (not clear)
Tsys = 135 #W named Ts is "Characterization of On-Orbit GPS"
Rm_km = 6371 #km
d_lagrange_km = np.array([570], dtype='int64')
R = 10**6 * 0.5 / (1/12) #bit/s, see GPSReceiverArchitecturesandMeasurements.
La_dB = -0.5

def Gpeak_helical(_D, _L, _lamba):
    return 10 * np.log10((np.pi**2 * _D**2 * _L)/(_lamba**3)) + 10.3 # dB

def Gpeak_parabolic_t(_D, _f):
    f = _f/(1e9)
    return 20*np.log10(_D) + 20* np.log10(f) + 17.8
def Gpeak_parabolic_r(_D,_eta,_lamba):
    return (np.pi**2 * _D **2 * _eta)/(_lamba**2)

def alpha1_over_2_helical(_D, _L, _lamba):
    return (52)/(np.sqrt((np.pi**2 * _D**2 * _L)/(_lamba**3)))

def alpha1_over_2_parabolic(_f, _D):
    return 21/((_f*10**-9)*_D)

def Lpr(_et, _alpha):
    return -12*(_et/_alpha)**2

def dB(_X):
    return 10*np.log10(_X)

def snr(_P_dB, _Ll_dB, _Gt_dB, _Gr_dB, _Ls_dB, _Lpr_dB, _Lr_dB, _R_dB, _k_dB, _Ts_dB): #no La, all values in dB
    return _P_dB+_Ll_dB+_Gt_dB+_Gr_dB+_Ls_dB+_Lpr_dB+_Lr_dB-_R_dB-_k_dB-_Ts_dB


def snr_with_La(_P_dB, _Ll_dB, _Gt_dB, _Gr_dB, _Ls_dB, _Lpr_dB, _Lr_dB, _R_dB, _k_dB, _Ts_dB, _La_dB): #no La, all values in dB
    return _P_dB+_Ll_dB+_Gt_dB+_Gr_dB+_Ls_dB+_Lpr_dB+_Lr_dB+_La_dB-_R_dB-_k_dB-_Ts_dB


def snr_margin(_SNR_required, _snr_actual):
    return _SNR_required - _snr_actual


def S_lagrange(_Rp, _d):       # Formula from slides, slightly modified for lagrange, as lagrange points or from center of the moon. this means Rm**2+h**2 = d**2
    return np.sqrt(_d**2 - _Rp**2)

def S(_Rp, _d):       # Formula from slides, slightly modified for lagrange, as lagrange points or from center of the moon. this means Rm**2+h**2 = d**2
    return np.sqrt((_Rp+_d)**2 - _Rp**2)

def L_s(_lamba, _S):
    return (_lamba/(4*np.pi*_S))**2


#helical
# alpha1_over_2_tr = alpha1_over_2_helical(Dt, L, lamba)
# Gt_dB = Gpeak_helical(Dt, L, lamba)
# Lpr_tr_dB = Lpr(et_tr, alpha1_over_2_tr)
# Lpr_re_dB = Lpr(0.1, 1)
# Lpr_tot_dB = Lpr_tr_dB + Lpr_re_dB


#parabolic
Gt_dB = Gpeak_parabolic_t(D_tr, f)
Gr = Gpeak_parabolic_r(D_r, eta, lamba)
Lpr_tr_dB = Lpr(et_tr, alpha1_over_2_parabolic(f, D_tr))
Lpr_re_dB = Lpr(0.1, 1)
Lpr_tot_dB = Lpr_tr_dB+Lpr_re_dB


#check to see if lagrange point or not.
S1 = S(Rm_km, d_lagrange_km)*1000 
# S1 = S_lagrange(Rm_km, d_lagrange_km)*1000 

Ls = L_s(lamba, S1)  # just a formula from slides

SNR_req = 10 # dB max that is possible

Ptr_dB = dB(Ptr)
Gr_dB = dB(Gr)
Ll_dB = dB(Ll)
Ls_dB = dB(Ls)
Lr_dB = dB(Lr)
R_dB = dB(R)
k_dB = dB(BOLTZMANN)
Tsys_dB = dB(Tsys)

SNR_act = snr_with_La(Ptr_dB, Ll_dB, Gt_dB, Gr_dB, Ls_dB, Lpr_tot_dB, Lr_dB, R_dB, k_dB, Tsys_dB, La_dB)

print(f'{Ptr_dB=}')
print(f'{Ll_dB=}')
print(f'{Gt_dB=}')
print(f'{La_dB=}')
print(f'{Gr_dB=}')
print(f'{Ls_dB=}')
print(f'{Lpr_tr_dB=}')
print(f'{Lpr_re_dB=}')
print(f'{Lpr_tot_dB=}')
print(f'{Lr_dB=}')
print(f'{R_dB=}')
print(f'{k_dB=}')
print(f'{Tsys_dB=}')

print(snr_margin(SNR_act, SNR_req))

#this proves that the 