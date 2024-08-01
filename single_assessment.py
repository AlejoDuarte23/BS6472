from bs_6473 import Service_assessment
from scipy.io import loadmat
from typing import List, Dict, Any
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
    Acc_x: Any
    Acc_y: Any
    Acc_z: Any


class VibrationSurvey(BaseModel):
    list_of_tests: List[VibrationTest]
    
    def append(self, test: VibrationTest):
        self.list_of_tests.append(test)

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
def plot_psd(acc: np.ndarray, fs: float):
    # Define nfft as 2**10
    nfft = 2**10
    
    # Compute the Power Spectral Density using Welch's method
    f, Pxx = welch(acc, fs, nperseg=nfft)
    
    # Plot the PSD
    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title('Power Spectral Density (PSD)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V^2/Hz)')
    plt.grid(True)
    plt.show()


def calculate_rms_of_peaks(acc: np.ndarray):
    # Find peaks in the signal
    peaks, _ = find_peaks(acc)
    
    # Get the values at the peaks
    peak_values = acc[peaks]
    
    # Calculate RMS of the peak values
    rms_peaks = np.sqrt(np.mean(peak_values**2))
    
    return rms_peaks

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
                                            rms_fix= 0.18)
    
    # curve factor as per BS 6472
    curve_factors = [1,2,4,8,24]
    labels = [f"{factor}x base curve" for factor in curve_factors]
    labels[0] = 'Base curve'
    
    # Assessment plot
    service_assessment.BS_6472(act_fact=[1,2,4,8,24], labels=labels)


    