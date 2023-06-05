"""This code takes two points and uses the link budget parameters between them to calculate the signal-to-noise ratio.
By J. Geijsberts"""

## references
# Tsys
# system temperature: first order is gained from smad: 135
# system temperature = 175.84 #W named Ts is "Characterization of On-Orbit GPS"

# External Libraries
import numpy as np

# Local Libraries

# Constants
BOLTZMANN = 1.38065E-23 
SPEED_OF_LIGHT = 299792458 #m/s
RADIUS_MOON = 1737500 #m
ANTENNA_1 = {'type': 'helical', 'diameter':0.098, 'length':0.212,} # efficiency must be added
ANTENNA_2 = {'type': 'parabolic', 'diameter': 3, 'length':0.5}  #TEST VALUE



#Functions
# def peakGainHelical(_diameter, _length, _wavelength):
#     return 10 * np.log10((np.pi**2 * _diameter**2 * _length)/(_wavelength**3)) + 10.3 # dB

def peakGainParabolicTransmitter(_diameter, _frequency):
    frequency = _frequency/(1e9)
    return 20 * np.log10(_diameter) + 20 * np.log10(frequency) + 17.8

def peakGainParabolicReceiver(_diameter,_eta,_wavelength):
    return (np.pi**2 * _diameter **2 * _eta)/(_wavelength**2)

def halfPowerAngleHelical(_diameter, _length, _wavelength):
    return (52)/(np.sqrt((np.pi**2 * _diameter**2 * _length)/(_wavelength**3)))

def halfPowerAngleParabolic(_diameter, _frequency):
    return 21/((_frequency*10**-9)*_diameter)

def pointingLoss(_et, _alpha):
    return -12*(_et/_alpha)**2

def spaceLoss(_wavelength, _distance):
    return (_wavelength/(4*np.pi*_distance))**2

def decibel(_X):
    return 10*np.log10(_X)

def largestDistance(_radius_planet, _distance):       # Formula from slides, slightly modified for lagrange, as lagrange points or from center of the moon. this means Rm**2+h**2 = d**2
    return np.sqrt((_radius_planet+_distance)**2 - _radius_planet**2)

# def calculateAntennaGain(_position, _antenna, _frequency, _wavelength):
#     if _antenna['type']=='helical':
#             return peakGainHelical(_antenna['diameter'], _antenna['length'], _wavelength)
#     elif _antenna['type']=='parabolic':
#         if _position == 'transmitter':
#             return peakGainParabolicTransmitter(_antenna['diameter'], _frequency)
#         if _position == 'receiver':
#             return peakGainParabolicReceiver(_antenna['diameter'], _antenna['efficiency'], _wavelength)
#     # elif _antenna['type']=='laser':
        # self.gain_transmitter = 'laser'
    # else:
    #     raise ValueError("This antenna type is invalid.")

# def calculatePointingLoss(_antenna, _frequency, _wavelength):
#     if _antenna['type']=='helical':
#         return halfPowerAngleHelical(_antenna['diameter'], _antenna['length'], _wavelength)
#     if _antenna['type']=='parabolic':
#         return halfPowerAngleParabolic(_antenna['diameter'], _frequency)
#     else:
#         raise ValueError("This antenna type is invalid.")


#Classes

class HelicalAntenna:
    def __init__(self, diameter, length):
        self.diameter = diameter
        self.length = length
    
    def peak_gain(self, wavelength):
        return 10 * np.log10((np.pi**2 * self.diameter**2 * self.length)/( wavelength ** 3 )) + 10.3
    
    def half_power_angle(self, wavelength):
        return (52)/(np.sqrt((np.pi**2 * self.diameter**2 * self.length)/(wavelength**3)))
    
class ParabolicAntenna:
    def __init__(self, diameter, efficiency):
        self.diameter = diameter
        self.efficiency = efficiency

    def peak_gain_transmitter(self, frequency):
        frequency = frequency / 1e9
        return 20 * np.log10(self.diameter) + 20 * np.log10(frequency) + 17.8

    def peak_gain_receiver(self, wavelength):
        return (np.pi ** 2 * self.diameter ** 2 * self.efficiency) / (wavelength ** 2)

    def half_power_angle(self, frequency):
        return 21 / ((frequency * 10 ** -9) * self.diameter)
    

class LinkBudget:
    def __init__(self, frequency, loss_factor_receiver, loss_factor_transmitter, power_transmitter, data_rate, atmospheric_loss, system_temperature, antenna_transmitter, antenna_receiver):
        """Creates a link budget between two points."""
        self._frequency = frequency
        self._snr_required = None
        self.wavelength = SPEED_OF_LIGHT/frequency
        self._power_transmitter = power_transmitter
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
        self.antenna_transmitter = antenna_transmitter
        self.antenna_receiver = antenna_receiver
        
    def calculateAntennaGain(self, antenna, position):
        if isinstance(antenna, HelicalAntenna):
            return antenna.peak_gain(self.wavelength)
        elif isinstance(antenna, ParabolicAntenna):
            if position == 'transmitter':
                return antenna.peak_gain_transmitter(self.frequency)
            elif position == 'receiver':
                return antenna.peak_gain_receiver(self.wavelength)
        else:
            raise ValueError("This antenna type is invalid.")
        
    def calculatePointingLoss(self, antenna):
        if isinstance(antenna, HelicalAntenna):
            return antenna.half_power_angle(self.wavelength)
        elif isinstance(antenna, ParabolicAntenna):
            
            return antenna.half_power_angle(self.frequency)
        else:
            raise ValueError("This antenna type is invalid.")
        
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
    def snr_required(self):
        return self._snr_required
    
    @snr_required.setter
    def snr_required(self, value):
        if value >=0:
            self._snr_required = value
        else:
            raise ValueError("the SNR required must be larger than 0.")
        
    @property
    def power_transmitter(self):
        return self._power_transmitter
    
    @power_transmitter.setter
    def power_transmitter(self, value):
        if value >0:
            self._power_transmitter = value
        else:
            raise ValueError("The power of the transmitting antenna must be larger than zero")

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
    def pointing_loss(self):
        return self._pointing_loss
    
    @pointing_loss.setter
    def pointing_loss(self, value):
        self._pointing_loss = value

    @property
    def antenna_transmitter(self):
        return self.antenna_transmitter
    
    @antenna_transmitter.setter
    def antenna_transmitter(self, value):
        position = 'transmitter'
        if value is None:
            raise ValueError("There is no transmitting antenna.")
                
        self.gain_transmitter = self.calculateAntennaGain(value, position)
        self.pointing_loss_transmitter = self.calculatePointingLoss(value) 
        self._antenna_transmitter = value      
        # if value['type']=='helical':
            #     self.gain_transmitter = peakGainHelical(value['diameter'], value['length'], self.wavelength)
            #     self.pointing_loss
            # elif value['type']=='parabolic':
            #     self.gain_transmitter = peakGainParabolicTransmitter(value['diameter'], self.frequency)
            # # elif value['type']=='laser':
            #     # self.gain_transmitter = 'laser'
            # else:
            #     raise ValueError("This transmitting antenna type does not exist.")   
    
    @property
    def antenna_receiver(self):
        return self.antenna_receiver
    
    @antenna_receiver.setter
    def antenna_receiver(self, value):
        position = 'receiver'
        if value is None:
            raise ValueError("There is no receiver antenna.")
        self.gain_receiver = self.calculateAntennaGain(value, position)
        self.pointing_loss_receiver = self.calculatePointingLoss(value)
        self._antenna_transmitter = value

if __name__=="__main__":
    # inputs: Ptr, Ll, Lr, k, frequency, char_transmitting_ant, char_receiving_ant, orbit/mission param(distance), tele-com requirements(data rate), coding-modulation type, required BER
    ant_1 = HelicalAntenna(diameter=0.098, length=0.212) # efficiency must be added
    ant_2 = ParabolicAntenna(3, 5)  #TEST VALUE


    a = LinkBudget(2500e6, None, None, 35, 50, None, None, ant_2, ant_1)
    print(f'{a.gain_receiver=}')
    print(f'{a.gain_transmitter=}')
    print(f'{a.pointing_loss_transmitter=}')