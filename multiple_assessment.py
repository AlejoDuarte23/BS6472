from bs_6473 import Service_assessment
from scipy.io import loadmat
from typing import List, Dict, Any, Literal
from pydantic import BaseModel
from numpy.typing import NDArray
from scipy.signal import welch, find_peaks

from utils import AlignMultipleMeasurements


import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.use('Qt5Agg')

#%% Clasess
class VibrationTest(BaseModel):
    name: str
    fs: float
    Acc_x: Any
    Acc_y: Any
    Acc_z: Any

    def set_accelerations(self, axis: Literal['Acc_x', 'Acc_y', 'Acc_z'], data: np.ndarray):
        setattr(self, axis, data)


class VibrationSurvey(BaseModel):
    list_of_tests: List[VibrationTest]
    
    def append(self, test: VibrationTest):
        self.list_of_tests.append(test)

    def get_axis_data(self, axis: Literal['Acc_x', 'Acc_y', 'Acc_z']) -> List[np.ndarray]:
        return [getattr(test, axis) for test in self.list_of_tests]

#%% Data
data_list = [
"data/grandstand_N02_force/NO-02_13DIC_cel5_D.csv",
"data/grandstand_N02_force/NO-02_13DIC_cel7_D.csv",
"data/grandstand_N02_force/NO-02_13DIC_cel8_D.csv",
"data/grandstand_N02_force/NO-02-13DIC-cel2-D.csv",
"data/grandstand_N02_force/NO-02-13DIC-Cel3-D.csv",
]

names = ["cel5", "cel7", "cel8", "cel2", "cel3"]

headers = ['time (sec)', 'X vibration (g)', 'Y vibration (g)', 'Z vibration (g)']

#%% Functions
def calculate_fs(data:NDArray) -> float:
    fs = 1/(np.mean(np.diff(data)))
    return fs

def align_data_from_vibration_survey(vibration_survey: VibrationSurvey, axis: Literal['Acc_x', 'Acc_y', 'Acc_z']) -> VibrationSurvey:
    # Align the data

    aligned_signals = AlignMultipleMeasurements([getattr(test, axis) for test in vibration_survey.list_of_tests]).align_signals()
    # # Update the data in the VibrationSurvey object
    for i, test in enumerate(vibration_survey.list_of_tests):
        test.set_accelerations(axis, aligned_signals[i])
        print(len(getattr(test, axis)))  # This will print the length of each aligned signal
    
    return vibration_survey
if __name__ == '__main__':
    cutoff = 10000
    vibration_survey = VibrationSurvey(list_of_tests=[])
    for data , name in zip(data_list, names):
        df_data = pd.read_csv(data)
        fs = calculate_fs(data = df_data[headers[0]].to_numpy()[:1000])
        vibration_test = VibrationTest(name = name, fs = fs, Acc_x = df_data[headers[1]].to_numpy()[cutoff:-1]
                                    , Acc_y = df_data[headers[2]].to_numpy()[cutoff:-1],
                                        Acc_z = df_data[headers[3]].to_numpy()[cutoff:-1])
        vibration_survey.append(vibration_test)

    #Allign data from vibration survey
    vibration_survey_align_z = align_data_from_vibration_survey(vibration_survey, 'Acc_z')



    for name_id,acc_z in enumerate(vibration_survey_align_z.get_axis_data('Acc_z')):
        plt.plot(acc_z,label = f"{names[name_id]} - Z axis",alpha = 0.3)

    plt.legend()
    plt.show()
    


