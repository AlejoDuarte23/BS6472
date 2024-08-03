#%% Imports 
from bs_6473 import Service_assessment
from scipy.io import loadmat
from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from numpy.typing import NDArray
from scipy.signal import welch, find_peaks



import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use('Qt5Agg')

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


#%% Data
data_list = [
"data/grandstand_N02_force/NO-02_13DIC_cel1_D.csv",
"data/grandstand_N02_force/NO-02_13DIC_cel5_D.csv",
"data/grandstand_N02_force/NO-02_13DIC_cel7_D.csv",
"data/grandstand_N02_force/NO-02_13DIC_cel8_D.csv",
"data/grandstand_N02_force/NO-02-13DIC-cel2-D.csv",
"data/grandstand_N02_force/NO-02-13DIC-Cel3-D.csv"
]

names = ["cel1", "cel5", "cel7", "cel8", "cel2", "cel3"]

headers = ['time (sec)', 'X vibration (g)', 'Y vibration (g)', 'Z vibration (g)']

#%% Functions
def calculate_fs(data:NDArray) -> float:
    fs = 1/(np.mean(np.diff(data)))
    return fs

if __name__ == '__main__':

    vibration_survey = VibrationSurvey(list_of_tests=[])
    for data , name in zip(data_list, names):
        df_data = pd.read_csv(data)
        fs = calculate_fs(data = df_data[headers[0]].to_numpy()[:1000])
        vibration_test = VibrationTest(name = name, fs = fs, Acc_x = df_data[headers[1]].to_numpy()
                                    , Acc_y = df_data[headers[2]].to_numpy(),
                                        Acc_z = df_data[headers[3]].to_numpy())
        vibration_survey.append(vibration_test)

    # Process the data
    acc_data = vibration_survey.list_of_tests[5].Acc_z[39600:42000]*9.81
    fs = vibration_survey.list_of_tests[5].fs
    _dir = 'Z'
    service_assessment = Service_assessment(acc_data=acc_data, 
                                            fs=fs, 
                                            _dir = _dir,
                                            rms= 0.18)
    
    # Curve factor as per BS 6472
    curve_factors = [1,2,4,8,24]
    labels = [f"{factor}x base curve" for factor in curve_factors]
    labels[0] = 'Base curve'
    
    service_assessment.BS_6472(act_fact=[1,2,4,8,24], labels=labels, tooltip=True , sensor_names=['Sensor 3'])


    