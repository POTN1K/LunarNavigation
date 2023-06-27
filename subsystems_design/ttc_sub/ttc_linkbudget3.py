"""This code takes two points and uses the link budget parameters between them to calculate the signal-to-noise ratio.
By J. Geijsberts"""

## references
# Tsys
# system temperature: first order is gained from smad: 135
# system temperature = 175.84 #W named Ts is "Characterization of On-Orbit GPS"

# External Libraries
import numpy as np
from functools import reduce

# Local Libraries

# Constants
BOLTZMANN = 1.38065E-23 
SPEED_OF_LIGHT = 299792458 #m/s
RADIUS_MOON = 1737500 #m
DATA_RATE_NAVIGATION_SIGNAL = 50 #b/s To user
DATA_RATE_TOTAL_SIGNAL = 1000 #b/s to both LNSS and relay

#Functions
def decibel(_X):
    if _X != 0:
        return 10*np.log10(_X)
    else:
        return 0

def inv_decibel(_X):
    return 10 ** (_X/10) 

def recursive_times(*args):
    return reduce(lambda x,y:x*y, args, 1)



class Transmitter:
    def __init__(self, gain, power, pointing_loss, other_losses):
        self._gain = gain
        self._power = power
        self._pointing_loss = pointing_loss
        self._other_losses = other_losses

    # def totalPowerTransmitter(self):
    #     args = [self.gain, self.power, self.losses]
    #     filtered = []
    #     for i in args:
    #         if isinstance(i, int or float):
    #             filtered.append(i)
    #     return recursive_times(*filtered)

    @property
    def gain(self):
        return self._gain    
    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def power(self):
        return self._power
    @power.setter
    def power(self, value):
        self._power = value

    @property
    def pointing_loss(self):
        return self._pointing_loss
    @pointing_loss.setter
    def pointing_loss(self, value):
        self._pointing_loss = value
    

    @property
    def other_losses(self):
        return self._other_losses
    @other_losses.setter
    def other_losses(self, value):
        self._other_losses = value
    
    
    @property
    def total_power_transmitter(self):
        args = [self.gain, self.power, self.pointing_loss, self.other_losses]
        filtered = []
        for i in args:
            if isinstance(i, int) or isinstance(i, float):
                filtered.append(i)
        return recursive_times(*filtered)


class Receiver:
    def __init__(self, gain, pointing_loss, other_losses):
        self.gain = gain
        self.pointing_loss = pointing_loss
        self.other_losses = other_losses

    @property
    def gain(self):
        return self._gain
    @gain.setter
    def gain(self, value):
        self._gain = value

    @property
    def other_losses(self):
        return self._other_losses
    @other_losses.setter
    def other_losses(self, value):
        self._other_losses = value
    
    @property
    def pointing_loss(self):
        return self._pointing_loss
    @pointing_loss.setter
    def pointing_loss(self, value):
        self._pointing_loss = value
        


    @property
    def total_gain_receiver(self):
        args = [self.gain, self.pointing_loss, self.other_losses]
        filtered = []
        for i in args:
            if isinstance(i, int) or isinstance(i, float):
                filtered.append(i)
        return recursive_times(*filtered)



class NoiseFigure:
    def __init__(self, boltzmann, data_rate, system_temperature):
        self.boltzmann = boltzmann
        self.data_rate = data_rate
        self.system_temperature = system_temperature
    
    @property
    def denominator(self):
        args = [self.boltzmann, self.data_rate, self.system_temperature]
        filtered = []
        for i in args:
            if isinstance(i, int) or isinstance(i, float):
                filtered.append(i)
        return recursive_times(*filtered)
    


class LinkBudget:
    def __init__(self, frequency, distance):
        self._snr = None
        self._snr_required = None
        self._transmitter = None
        self._receiver = None
        self._noise_figure = None
        self._space_loss = None
        self.frequency = frequency
        self.distance = distance


    @property
    def space_loss(self):
        return ((SPEED_OF_LIGHT/self.frequency)/(4*np.pi*self.distance))**2

    @property
    def snr_required(self):
        return self._snr_required
    
    @property
    def transmitter(self):
        return self._transmitter
    
    @transmitter.setter
    def transmitter(self, value):
        self._transmitter = value
    
    @property
    def receiver(self):
        return self._receiver

    @receiver.setter
    def receiver(self, value):
        self._receiver = value

    @property
    def noise_figure(self):
        return self._noise_figure

    @noise_figure.setter
    def noise_figure(self, value):
        self._noise_figure = value

    @property
    def space_loss(self):
        return ((SPEED_OF_LIGHT / self.frequency)/(4*np.pi*self.distance))**2




a = LinkBudget(2492e6, 10466240)

a.transmitter = Transmitter(None, None, inv_decibel(-0.002657312925170068), None)
# a.receiver = Receiver(inv_decibel(3.05), inv_decibel(-0.12000000000000002), None)
a.noise_figure = NoiseFigure(BOLTZMANN, 50, 135)
min_power_rec = -160
total = min_power_rec - decibel(a.space_loss) - 8 - 3.05
print('the power is', inv_decibel(total), 'Watts')


# total_pt = inv_decibel(8) * a.noise_figure.denominator / (a.space_loss * a.receiver.total_gain_receiver)
# total_pt = total_pt/(inv_decibel(-0.002657312925170068) * inv_decibel(8))
# print(f'{total_pt=}')




