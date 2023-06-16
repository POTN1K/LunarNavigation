"""All constants for structure sizing are stored here.
By I. Maes, N. Ricker"""



axial_freq_falcon = 25  # [Hz] Axial freq of the Falcon Heavy
lateral_freq_falcon = 10  # [Hz] Lateral freq of the Falcon Heavy
g_axial = 8.5   # [g] Axial load factor
g_lateral = 3   # [g] Lateral load factor
g_tensile = 4   # [g] Tensile load factor (negative g)
g = 9.80665
temperatures = [0, 40]

material_properties = {
    "Aluminium_6061-T6": {
        "density": 2710,  # [kg/m^3]
        "yield_strength": 276e6,  # [Pa]
        "ultimate_strength": 310e6,  # [Pa]
        "E": 68.9e9,  # [Pa]
        "thermal_coefficient": 23.6e-6,  # [m/m]
    },
    "Aluminium_7075-T73": {
        "density": 2800,  # [kg/m^3]
        "yield_strength": 435e6,  # [Pa]
        "ultimate_strength": 505e6,  # [Pa]
        "E": 72e9,  # [Pa]
        "thermal_coefficient": 23.6e-6,  # [m/m]
    },
    "Aluminium_2219-T851": {
        "density": 2850,  # [kg/m^3]
        "yield_strength": 352e6,  # [Pa]
        "ultimate_strength": 455e6,  # [Pa]
        "E": 73.1e9,  # [Pa]
        "thermal_coefficient": 22.3e-6,  # [m/m]
    },
    "Ti-6AL-4V": {
        "density": 4430,  # [kg/m^3]
        "yield_strength": 880e6,  # [Pa]
        "ultimate_strength": 950e6,  # [Pa]
        "E": 113.8e9,  # [Pa]
        "thermal_coefficient": 8.6e-6,  # [m/m]
    },
    "Magnesium": {
        "density": 1770,  # [kg/m^3]
        "yield_strength": 220e6,  # [Pa]
        "ultimate_strength": 290e6,  # [Pa]
        "E": 45e9,  # [Pa]
        "thermal_coefficient": 26e-6,  # [m/m]
    },
    "Heat-res_alloy_A-286": {
        "density": 7940,  # [kg/m^3]
        "yield_strength": 720e6,  # [Pa]
        "ultimate_strength": 1000e6,  # [Pa]
        "E": 201e9,  # [Pa]
        "thermal_coefficient": 0,  # [m/m]
    },
    "Heat-res_alloy_inconel_718": {
        "density": 8220,  # [kg/m^3]
        "yield_strength": 1034e6,  # [Pa]
        "ultimate_strength": 1241e6,  # [Pa]
        "E": 200e9,  # [Pa]
        "thermal_coefficient": 0,  # [m/m]
    },
    "Steel_17-4PH_H1150": {
        "density": 7860,  # [kg/m^3]
        "yield_strength": 862e6,  # [Pa]
        "ultimate_strength": 1000e6,  # [Pa]
        "E": 196e9,  # [Pa]
        "thermal_coefficient": 0  # [m/m]
    },
    "Beryllium": {
        "density": 1850,  # [kg/m^3]
        "yield_strength": 241e6,  # [Pa]
        "ultimate_strength": 324e6,  # [Pa]
        "E": 290e9,  # [Pa]
        "thermal_coefficient": 11.3  # [m/m]
    }}

components = {
    "cmg1": {
        "subsystem": "ADCS",
        "mass": 10,  # [kg]
        "cg": [0, 0.37, 0]  # [m]
    },
    "cmg2": {
        "subsystem": "ADCS",
        "mass": 10,  # [kg]
        "cg": [0, -0.37, 0]  # [m]
    },
    "star_sensors": {
        "subsystem": "ADCS",
        "mass": 4*0.47,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "sun_sensors": {
        "subsystem": "ADCS",
        "mass": 15*0.05,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "ring_laser_gyros": {
        "subsystem": "ADCS",
        "mass": 4*0.454,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "computer": {
        "subsystem": "CDH",
        "mass": 9,  # [kg]
        "cg": [-.35, -0.3, .375]  # [m]
    },
    "Galileo_PCDU": {
        "subsystem": "EPS",
        "mass": 18.2,  # [kg]
        "cg": [-.175, 0.400, -.200]  # [m]
    },
    "NSGU": {
        "subsystem": "Navigation",
        "mass": 12,  # [kg]
        "cg": [0.35, 0, .360]  # [m]
    },
    "FGUU": {
        "subsystem": "Navigation",
        "mass": 7.6,  # [kg]
        "cg": [-0.1, 0, .360]  # [m]
    },
    "Clock_Monitor": {
        "subsystem": "Navigation",
        "mass": 5.2,  # [kg]
        "cg": [0.05, -.400, -0.4]  # [m]
    },
    "Clock1": {
        "subsystem": "Navigation",
        "mass": 15.9,  # [kg]
        "cg": [.350, -0.3, 0.36]  # [m]
    },
    "Clock2": {
        "subsystem": "Navigation",
        "mass": 15.9,  # [kg]
        "cg": [.350, 0.3, 0.36]  # [m]
    },
    "Clock3": {
        "subsystem": "Navigation",
        "mass": 15.9,  # [kg]
        "cg": [-.350, 0.3, 0.36]  # [m]
    },
    "Battery1": {
        "subsystem": "EPS",
        "mass": 48,  # [kg]
        "cg": [0, 0, -0.38]  # [m]
    },
    "Battery2": {
        "subsystem": "EPS",
        "mass": 29.64,  # [kg]
        "cg": [0, 0.4, -0.37]  # [m]
    },
    "Cables": {
        "subsystem": "EPS",
        "mass": 100,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "GR1_Thrusters": {
        "subsystem": "Propulsion",
        "mass": 12*0.29,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "GR22_Thruster": {
        "subsystem": "Propulsion",
        "mass": 0.59*2,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Filters_Valves": {
        "subsystem": "Propulsion",
        "mass": 4.4,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "TTC_User": {
        "subsystem": "TTC",
        "mass": 4*0.6278,  # [kg]
        "cg": [0, 0, 0.5]  # [m]
    },
    "TTC_Relay": {
        "subsystem": "TTC",
        "mass": 0.19085,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "TTC_ISL": {
        "subsystem": "TTC",
        "mass": 4,  # [kg]
        "cg": [-0.6, 0, -0.27]  # [m]
    },
    "TTC_Reflector": {
        "subsystem": "TTC",
        "mass": 12.43,  # [kg]
        "cg": [-0.6, 0, 0.23]  # [m]
    },
    "Radiator": {
        "subsystem": "TCS",
        "mass": 11.85,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Heaters": {
        "subsystem": "TCS",
        "mass": 1,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Phase_Change_Material": {
        "subsystem": "TCS",
        "mass": 35,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Antenna_Support": {
        "subsystem": "Structures",
        "mass": 3,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Mechanisms": {
        "subsystem": "Structures",
        "mass": 10,  # [kg]
        "cg": [0, 0, 0]  # [m]
    },
    "Tank_Support": {
        "subsystem": "Structures",
        "mass": 10,  # [kg]
        "cg": [0, 0, 0]  # [m]
    }
}
