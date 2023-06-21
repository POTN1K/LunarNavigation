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
# ANTENNA_1 = {'type': 'helical', 'diameter':0.098, 'length':0.212,} # efficiency must be added
# ANTENNA_2 = {'type': 'parabolic', 'diameter': 3, 'length':0.5}  #TEST VALUE


#Functions
def decibel(_X):
    if _X != 0:
        return 10*np.log10(_X)
    else:
        return 0

def inv_decibel(_X):
    return 10 ** (_X/10) 

#Classes

class HelicalAntenna:
    def __init__(self, diameter, length, gain, power_transmitter):
        self.diameter = diameter
        self.length = length
        self.gain = gain
        self.power_transmitter = power_transmitter
    
    
    def peak_gain(self, wavelength):
        return inv_decibel(10 * np.log10((np.pi**2 * self.diameter**2 * self.length)/( wavelength ** 3 )) + 10.3)
    
    def half_power_angle(self, wavelength):
        return (52)/(np.sqrt((np.pi**2 * self.diameter**2 * self.length)/(wavelength**3)))
    

class ParabolicAntenna:
    def __init__(self, diameter, efficiency, gain, power_transmitter):
        self.diameter = diameter
        self.efficiency = efficiency
        self.gain = gain
        self.power_transmitter = power_transmitter

    def peak_gain_transmitter(self, frequency):
        frequency = frequency / 1e9
        return inv_decibel(20 * np.log10(self.diameter) + 20 * np.log10(frequency) + 17.8)

    def peak_gain_receiver(self, wavelength):
        return (np.pi ** 2 * self.diameter ** 2 * self.efficiency) / (wavelength ** 2)

    def half_power_angle(self, frequency):
        return 21 / ((frequency * 10 ** -9) * self.diameter)
    

class LaserAntenna:
    def __init__(self, efficiency, diameter, pointing_error, gain, power_transmitter):
        self.efficiency = efficiency
        self.diameter = diameter
        self.pointing_error = np.array([pointing_error], dtype='int64')
        self.gain = gain
        self.power_transmitter = power_transmitter

    def gain_laser(self, wavelength):
        return ((np.pi * self.diameter)/(wavelength))**2
    
    def pointing_loss(self, wavelength):
        return np.e**(-self.gain_laser(wavelength) * self.pointing_error**2 )
    


class LinkBudget:
    def __init__(self, frequency, loss_factor_receiver, loss_factor_transmitter, data_rate, atmospheric_loss, system_temperature, antenna_transmitter, antenna_receiver, distance, pointing_offset_angle_transmitter, pointing_offset_angle_receiver, snr_required_decibel, bandwidth):
        """Creates a link budget between two points."""
        self._frequency = np.array([frequency], dtype='int64')
        self._snr_required_decibel = snr_required_decibel
        self._optical_efficiency_transmitter = None
        self._optical_efficiency_receiver = None
        self.wavelength = SPEED_OF_LIGHT/frequency
        self._power_transmitter = None
        self._loss_factor_receiver = loss_factor_receiver
        self._loss_factor_transmitter = loss_factor_transmitter
        self._data_rate = data_rate
        self._atmospheric_loss = atmospheric_loss
        self._system_temperature = system_temperature
        self._gain_transmitter = None
        self._gain_receiver = None
        self._pointing_loss_transmitter = None
        self._pointing_loss_receiver = None
        self._pointing_loss = None
        self._space_loss = None
        self._snr_margin = None
        self._distance = distance
        self.boltzmann = BOLTZMANN
        self._half_power_angle_transmitter = None
        self._half_power_angle_receiver = None
        self._pointing_offset_angle_transmitter = pointing_offset_angle_transmitter
        self._pointing_offset_angle_receiver = pointing_offset_angle_receiver
        self.antenna_transmitter = antenna_transmitter
        self.antenna_receiver = antenna_receiver
        self.bandwidth = bandwidth
        
    def calculateAntennaGain(self, antenna, position):
        if isinstance(antenna, HelicalAntenna):
            return antenna.peak_gain(self.wavelength)
        elif isinstance(antenna, ParabolicAntenna):
            if position == 'transmitter':
                return antenna.peak_gain_transmitter(self.frequency)
            elif position == 'receiver':
                return antenna.peak_gain_receiver(self.wavelength)
        elif isinstance(antenna, LaserAntenna):
            pass
        else:
            raise ValueError("This antenna type is invalid.")
        
    def calculateHalfPowerAngle(self, antenna):
        if isinstance(antenna, HelicalAntenna):
            return antenna.half_power_angle(self.wavelength)
        elif isinstance(antenna, ParabolicAntenna):
            return antenna.half_power_angle(self.frequency)
        elif isinstance(antenna, LaserAntenna):
            pass
        else:
            raise ValueError("This antenna type for calculating half power angle does not exist.")
    
    def pointingLoss(self, _pointing_offset_angle, _half_power_angle):
        return inv_decibel(-12*(_pointing_offset_angle/_half_power_angle)**2)
    
    # def spaceLoss(self, _wavelength, _distance):
    #     return (_wavelength/(4*np.pi*_distance))**2
    
    def largestDistance(self, _radius_planet, _distance):       # Formula from slides, slightly modified for lagrange, as lagrange points or from center of the moon. this means Rm**2+h**2 = d**2
        return np.sqrt((_radius_planet+_distance)**2 - _radius_planet**2)
    
    def calculateSignalToNoiseRatio(self, **kwargs): # not finished yet.
        negative_vars = ['boltzmann', 'data_rate', 'system_temperature']
        total = 0
        for var_name in kwargs:
            if kwargs[var_name] is not None:
                if var_name in negative_vars:
                    total -= decibel(kwargs[var_name])
                else:
                    total += decibel(kwargs[var_name])
        return total
    
    def getPowerTransmitter(self,  antenna):
        if isinstance(antenna, ParabolicAntenna or HelicalAntenna or LaserAntenna):
            return antenna.power_transmitter
    
    def powerNoise(self, boltzmann, noise_temperature, bandwidth):
        return boltzmann * noise_temperature * bandwidth
    
    
    @property
    def optical_efficiency_transmitter(self):
        return self._optical_efficiency_transmitter
    
    @optical_efficiency_transmitter.setter
    def optical_efficiency_transmitter(self, value):
        self._optical_efficiency_transmitter = value

    @property
    def optical_efficiency_receiver(self):
        return self._optical_efficiency_receiver
    
    @optical_efficiency_receiver.setter
    def optical_efficiency_receiver(self, value):
        self._optical_efficiency_receiver = value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if value<0:
            raise ValueError("The distance must be larger than 0.")
        self._distance = value

    @property
    def frequency(self):
        return self._frequency
    
    @frequency.setter
    def frequency(self, value):
        if value >=0:
            self._frequency = value
        else:
            raise ValueError("the frequency must be larger than 0.")

    @property
    def snr_required_decibel(self):
        return self._snr_required_decibel
    
    @snr_required_decibel.setter
    def snr_required_decibel(self, value):
        if value >=0:
            self._snr_required_decibel = value
        else:
            raise ValueError("the SNR required must be larger than 0.")
        
    @property
    def power_transmitter(self):
        return self._power_transmitter
    
    @power_transmitter.setter
    def power_transmitter(self, value):
        if value is not None:
            if value>0:
                self._power_transmitter = value
            else:
                raise ValueError("The power of the transmitting antenna must be larger than zero")
        else:
            pass
    @property
    def pointing_offset_angle_transmitter(self):
        return self._pointing_offset_angle_transmitter
    
    @pointing_offset_angle_transmitter.setter
    def pointing_offset_angle_transmitter(self, value):
        if value>0:
            self._pointing_offset_angle_transmitter = value
        else:
            raise ValueError("Pointing offset angle for the transmitter must be more than 0.")

    @property
    def pointing_offset_angle_receiver(self):
        return self._pointing_offset_angle_receiver
    
    @pointing_offset_angle_receiver.setter
    def pointing_offset_angle_receiver(self, value):
        if value>0:
            self._pointing_offset_angle_receiver = value
        else:
            raise ValueError("Pointing offset angle for the receiver must be more than 0.")

    @property
    def loss_factor_transmitter(self):
        return self._loss_factor_transmitter
    
    @loss_factor_transmitter.setter
    def loss_factor_transmitter(self, value):
        if value>=0 and value<=1:
            self._loss_factor_transmitter = value
        else:
            raise ValueError("The loss factor of the transmitter must be larger than 0 and lower than or equal to 1.")

    @property
    def loss_factor_receiver(self):
        return self._loss_factor_receiver
    
    @loss_factor_receiver.setter
    def loss_factor_receiver(self, value):
        if value>=0 and value<=1:
            self._loss_factor_receiver = value
        else:
            raise ValueError("The loss factor of the receiver must be larger than 0 and lower than or equal to 1.")
    
    @property
    def data_rate(self):
        return self._data_rate

    @data_rate.setter
    def data_rate(self, value):
        if value>0:
            self._data_rate = value
        else:
            raise ValueError("The data rate must be larger than 0.")
        
    @property
    def system_temperature(self):
        return self._system_temperature

    @system_temperature.setter
    def system_temperature(self, value):
        if value>0:
            self._system_temperature = value
        else:
            raise ValueError("The system temperature must be larger than 0.")
        
    @property
    def atmospheric_loss(self):
        return self._atmospheric_loss

    @atmospheric_loss.setter
    def atmospheric_loss(self, value):
        if value>0:
            self._atmospheric_loss = value
        else:
            raise ValueError("The atmospheric loss must be larger than 0.")

    @property
    def gain_transmitter(self):
        return self._gain_transmitter
    
    @gain_transmitter.setter
    def gain_transmitter(self, value):
        self._gain_transmitter = value
 
    @property
    def gain_receiver(self):
        return self._gain_receiver
    
    @gain_receiver.setter
    def gain_receiver(self, value):
        self._gain_receiver = value

    @property
    def pointing_loss_transmitter(self):
        return self._pointing_loss_transmitter
    
    @pointing_loss_transmitter.setter
    def pointing_loss_transmitter(self, value):
        self._pointing_loss_transmitter = value

    @property
    def pointing_loss_receiver(self):
        return self._pointing_loss_receiver
    
    @pointing_loss_receiver.setter
    def pointing_loss_receiver(self, value):
        self._pointing_loss_receiver = value

    
    @property
    def antenna_transmitter(self):
        return self.antenna_transmitter
    
    @antenna_transmitter.setter
    def antenna_transmitter(self, value):
        position = 'transmitter'
        self.power_transmitter = self.getPowerTransmitter(value)
        if value is None:
            raise ValueError("There is no transmitting antenna.")
        if isinstance(value, HelicalAntenna or ParabolicAntenna):
            self.gain_transmitter = self.calculateAntennaGain(value, position)
            self.half_power_angle_transmitter = self.calculateHalfPowerAngle(value) 
            self.pointing_loss_transmitter = self.pointingLoss(self.pointing_offset_angle_transmitter, self.calculateHalfPowerAngle(value))
        elif isinstance(value, LaserAntenna):
            self._optical_efficiency_transmitter = value.efficiency
            self._gain_transmitter = value.gain_laser(self.wavelength)
            self.pointing_loss_transmitter = value.pointing_loss(self.wavelength)
        self._antenna_transmitter = value      

    
    @property
    def antenna_receiver(self):
        return self.antenna_receiver
    
    @antenna_receiver.setter
    def antenna_receiver(self, value):
        position = 'receiver'
        if value is None:
            raise ValueError("There is no receiver antenna.")
        if isinstance(value, HelicalAntenna or ParabolicAntenna):
            self.gain_receiver = self.calculateAntennaGain(value, position)
            self.half_power_angle_receiver = self.calculateHalfPowerAngle(value)
            self.pointing_loss_receiver = self.pointingLoss(self.pointing_offset_angle_receiver, self.calculateHalfPowerAngle(value))
        elif isinstance(value, LaserAntenna):
            self._optical_efficiency_receiver = value.efficiency
            self._gain_receiver = value.gain_laser(self.wavelength)
            self.pointing_loss_receiver = value.pointing_loss(self.wavelength)
        self._antenna_transmitter = value

    @property
    def pointing_loss(self):
        return self.pointing_loss_transmitter * self.pointing_loss_receiver
    
    @property
    def space_loss(self):
        # return self.spaceLoss(self.wavelength, self.distance)
        # return self.spaceLoss(self.wavelength, self.largestDistance(RADIUS_MOON, self.distance))
        return (self.wavelength/(4*np.pi*self.distance))**2

        
    # @property # should probably use kwargs here if possible
    # def power_received(self):
    #     return self.power_transmitter * self.optical_efficiency_transmitter * self.optical_efficiency_receiver * self.gain_transmitter * self.gain_receiver * self.pointing_loss_transmitter * self.pointing_loss_receiver * self.space_loss
    
    @property
    def power_received(self, *args):
        return reduce(lambda x,y : x*y, args, 1)
    

    @property
    def power_noise(self):
        return self.
    # @property #maybe check if necessary. # use kwargs here:
    # def snr_margin(self):
    #     power_transmitter = self.power_transmitter
    #     loss_factor_transmitter = self.loss_factor_transmitter
    #     loss_factor_receiver = self.loss_factor_receiver
    #     gain_transmitter = self.gain_transmitter
    #     gain_receiver = self.gain_receiver
    #     atmospheric_loss = self.atmospheric_loss
    #     space_loss = self.space_loss
    #     pointing_loss = self.pointing_loss
    #     data_rate = self.data_rate
    #     boltzmann = self.boltzmann
    #     system_temperature = self.system_temperature
    #     return self.calculateSignalToNoiseRatio(power_transmitter=power_transmitter, loss_factor_transmitter=loss_factor_transmitter, loss_factor_receiver=loss_factor_receiver, gain_transmitter=gain_transmitter, gain_receiver=gain_receiver, atmospheric_loss=atmospheric_loss, space_loss=space_loss, pointing_loss=pointing_loss, data_rate=data_rate, boltzmann=boltzmann, system_temperature=system_temperature) - self.snr_required_decibel


if __name__=="__main__":
    # ant_1 = HelicalAntenna(diameter=0.098, length=0.212) # efficiency must be added
    # ant_2 = ParabolicAntenna(3, 0.55)  #TEST VALUE
    # a = LinkBudget(2500e6, 0.5, 0.5, 31.622777, 50, 0, 175.84, ant_1, ant_2,  np.array([24438000], dtype='int64'), 13, 50, 10)
    # a.gain_receiver = 1.047129 #(have to make the gain 0.2 dB for same value so this is 0.2 dB in watts)
    # a.pointing_loss_receiver = inv_decibel(-0.12) #(assumed value from slides, as 0.1 e_re is assumed)

    # print(f'{a.snr_margin=}')

    optical_antenna_1 = LaserAntenna(0.8, 0.08, 1e-6, None, 0.68953)
    optical_antenna_2 = LaserAntenna(0.8, 1, 1e-6, None, None)
    b = LinkBudget(SPEED_OF_LIGHT/1550e-9, None, None, None, None, None, optical_antenna_1, optical_antenna_2, 4500000, None, None,-65.5)
    print(f'{decibel(b.space_loss)=}')
    print(f'{b.power_transmitter=}')
    print(f'{b.optical_efficiency_transmitter=}')
    print(f'{b.optical_efficiency_receiver=}')
    print(f'{b.gain_transmitter=}')
    print(f'{b.gain_receiver=}')
    print(f'{b.pointing_loss_transmitter=}')
    print(f'{b.pointing_loss_receiver=}')
    print('snr margin', decibel(b.power_received/inv_decibel(b.snr_required_decibel)))
    
# power transmitter a little, gain transmitter, :

# todo 1: get something for coding methods to introduce snr required, or other ways to get snr required.
# todo 2: finish the snr_margin calculator.
# inputs: Ptr, Ll, Lr, k, frequency, char_transmitting_ant, char_receiving_ant, orbit/mission param(distance), tele-com requirements(data rate), coding-modulation type, required BER
