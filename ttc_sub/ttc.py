"""This code calculates the first order mass, power and volume of the TT&C system using first order relations"""

#External libraries
import numpy as np


#Constants
rho_antenna = 6.5 #kg/m^2 (this is from adsee reader. value range is 5-8 kg/m^2, applies for parabolic)

def mass_antenna_parabolic(rho, area):
    return rho * area

# for patch antennas up to 10W
# mass<80g
# size: 82x82x20mm 

def mass_transceiver(power, specific_power): #this is only for antennas up to 10 W
    return power/specific_power


def mass_amplifier_TWTA(power):
    return 0.07*power +0.634