from pydantic import BaseModel
from numpy.typing import NDArray
from bs_6473 import Service_assessment
from typing import List, Any, Literal
from utils import AlignMultipleMeasurements

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

# For interactive terminal
matplotlib.use('Qt5Agg')
matplotlib.pyplot.ion()

#%% Clasess
class VibrationTest(BaseModel):
    name: str
    fs: float
    Acc_x: np.ndarray
    Acc_y: np.ndarray
    Acc_z: np.ndarray
    
    def set_accelerations(self, axis: Literal['Acc_x', 'Acc_y', 'Acc_z'], data: np.ndarray):
        setattr(self, axis, data)
    
    class Config:
        arbitrary_types_allowed = True

class VibrationSurvey(BaseModel):
    list_of_tests: List[VibrationTest]
    
    def append(self, test: VibrationTest):
        self.list_of_tests.append(test)

    def get_axis_data(self, axis: Literal['Acc_x', 'Acc_y', 'Acc_z']) -> List[np.ndarray]:
        return [getattr(test, axis) for test in self.list_of_tests]

#%% Data and labels
data_list = [
    'data/grandstand_N04_force/NC-04_13DIC_cel1_D.csv',
    'data/grandstand_N04_force/NC-04_13DIC_cel2_D.csv',
    'data/grandstand_N04_force/NC-04_13DIC_cel4_D.csv',
    'data/grandstand_N04_force/NC-04_13DIC_cel7_D.csv',
    'data/grandstand_N04_force/NC-04-13DIC-Cel5–D.csv',
]

names = ["Sensor 1", "Sensor 2", "Sensor 4", "Sensor 7", "Sensor 5"]

headers = ['time (sec)', 'X vibration (g)', 'Y vibration (g)', 'Z vibration (g)']

#%% Functions
def calculate_fs(data:NDArray) -> float:
    fs = 1/(np.mean(np.diff(data)))
    return fs

def align_data_from_vibration_survey(vibration_survey: VibrationSurvey, axis: Literal['Acc_x', 'Acc_y', 'Acc_z']) -> VibrationSurvey:
    aligned_signals = AlignMultipleMeasurements([getattr(test, axis) for test in vibration_survey.list_of_tests]).align_signals()
    for i, test in enumerate(vibration_survey.list_of_tests):
        test.set_accelerations(axis, aligned_signals[i])
    return vibration_survey

def calculate_rms(data: np.array) -> float:
    return np.sqrt(np.mean(data**2))

def slice_array(data: np.array, start: int, end: int) -> np.array:
    return data[start:end]

#%% Main
if __name__ == '__main__':

    # Preprocess data 
    cutoff = 12000
    vibration_survey = VibrationSurvey(list_of_tests=[])
    for data , name in zip(data_list, names):
        df_data = pd.read_csv(data)
        fs = calculate_fs(data = df_data[headers[0]].to_numpy()[:1000])
        vibration_test = VibrationTest(name = name, fs = fs, Acc_x = df_data[headers[1]].to_numpy()[cutoff:-1]*9.81,
                                        Acc_y = df_data[headers[2]].to_numpy()[cutoff:-1]*9.81,
                                        Acc_z = df_data[headers[3]].to_numpy()[cutoff:-1]*9.81)
        vibration_survey.append(vibration_test)

    # Allign data from vibration survey
    vibration_survey_align_z = align_data_from_vibration_survey(vibration_survey, 'Acc_z')

    acc_z_al = vibration_survey_align_z.get_axis_data('Acc_z')
    acc_z = [slice_array(acc,0,5000) for acc in acc_z_al]

    rms_acc_z = [calculate_rms(acc) for acc in acc_z]
    max_index = np.argmax(rms_acc_z)

    # Tooltip asignation to the max rms value
    bool_list = [False] * len(data)
    bool_list[max_index] = True

    _dir = 'Z'
    service_assessment = Service_assessment(acc_data=acc_z, 
                                            fs=fs, 
                                            _dir = _dir,
                                            rms= rms_acc_z)
    
    # Curve factor as per BS 6472
    curve_factors = [1,2,4,8,24]
    labels = [f"{factor}x base curve" for factor in curve_factors]
    labels[0] = 'Base curve'
    
    # Assessment plot
    service_assessment.BS_6472(act_fact=[1,2,4,8,24], labels=labels,tooltip=bool_list,sensor_names=names)

    


