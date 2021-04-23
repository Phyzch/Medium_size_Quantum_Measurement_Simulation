import numpy as np
from Detector_class import detector

class full_system():

    def __init__(self ,Detector_1_parameter, Detector_2_parameter):
        dof1, frequency1, nmax1, initial_state1, energy_window1 = Detector_1_parameter
        dof2, frequency2, nmax2, initial_state2, energy_window2 = Detector_2_parameter

        self.detector1 = detector(dof1,frequency1,nmax1,initial_state1,energy_window1)
        self.detector2 = detector(dof2, frequency2, nmax2, initial_state2, energy_window2)
