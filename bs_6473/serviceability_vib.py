
import matplotlib.pyplot as plt
import numpy as np
import sys 
import json 


from pathlib import Path
from scipy.signal import find_peaks
from pathlib import Path
from typing import TypeVar, Union, List, overload, Literal , Tuple
from matplotlib.colors import LinearSegmentedColormap


def load_json(filepath: str):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return data

class Colormap:
    def __init__(self,colors: List[str] =  ["#4B0082", "#708090", "#2F4F4F"],
                 n_bins: int = 10,
                cmap_name: str = "blue_mate"):
         
        self.cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def get_color(self, index, total):
        """Get a specific color from the colormap based on index and total number of items."""
        return self.cm(index / (total - 1) if total > 1 else 0)



class Service_assessment():
    def __init__(self,
                 acc_data: np.array,
                 fs: float,
                 _dir = Literal["X", "Y", "Z"],
                 rms = Union[float,List[float]]):
        
        self.data = acc_data 
        self.fs = fs
        self._dir = _dir
        self.rms = rms
        self.ylimits = []
        self.xlimits = []
        self.x_ticks = [1, 2, 3, 5, 6.3, 10, 16, 25, 40, 63, 80]
        self.line_styles = [
                (5, 5), (10, 5), (15, 5), (20, 5), (25, 5),
                (5, 10), (5, 15), (5, 20), (5, 25), (10, 10)
            ]

    
    def get_weights(self)-> Tuple[List[float], List[float]]:
        '''Get the weighting factors for the Z axis based on BS 6472'''
        file_path = Path(__file__).parent / 'BS_6472_weights.json'
        data = load_json(filepath=file_path)
        frequency = data["BS_6472_weights_Z"]["frequency"]
        weights = data["BS_6472_weights_Z"]["weight"]
        return frequency, weights

    def get_acc(self, act_fact: int)-> Tuple[List[float], List[float]]:
        ''' Get the acceleration limits based on the activity factor'''
        file_path = Path(__file__).parent / 'BS_6472_curves.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
        frequency = data[f"BS_6472_accelaration_{self._dir}"]["frequency"]
        accelerations = data[f"BS_6472_accelaration_{self._dir}"]["accelerations"]
        # Factoring the accelerations base on activities 
        accelerations = [value * act_fact for value in accelerations]
        self.ylimits = [accelerations[0], accelerations[-1]]
        self.xlimits = [frequency[0], frequency[-1]]
        return frequency,accelerations
    
    def BS_6472(self, act_fact: Union[int, List[int]],
                labels: Union[str,List[str]],
                tooltip: Union[list,List[bool]],
                sensor_names: Union[str,list]) -> None:
    
        self.tooltip = tooltip
        self.sensor_names = sensor_names

        if isinstance(act_fact, int):
            act_fact = [act_fact]

        if isinstance(labels, str):
            labels = [labels]
        
        if isinstance(self.rms, float):
            self.data = [self.data]
            self.rms = [self.rms]
            self.tooltip = [True]
            self.sensor_names = [sensor_names]

        # Plot accelertion limits
        fig, ax = plt.subplots()
        colormap = Colormap()
        accel_limit_cmap = Colormap(colors = ["#FF0000", "#990000","#800000"],n_bins=6, cmap_name='red_mate')
        ax, accel_limits_list = self.plot_acceleration_limits(ax = ax,colormap = accel_limit_cmap,
                                      act_fact = act_fact, labels = labels)

        # Plot the weighted acceleration values in frequency domain        
        number_of_signals = len(self.data)
        for index, (signal, rms, tooltip_val,sensor_name) in enumerate(zip(self.data, self.rms, self.tooltip , self.sensor_names)):    
            self.plot_fft_rms(ax, signal, rms, color=colormap.get_color(index, number_of_signals),
                              tooltip=tooltip_val,sensor_name = sensor_name)

        # Add legend for acceleration limits 
        lines = ax.get_lines()
        legend1 = plt.legend([lines[i]for i in range(len(accel_limits_list))], labels, loc='lower right', framealpha=0.8)
        ax.add_artist(legend1)
        
        # Add legend for sensor names
        legend2 = plt.legend([lines[i] for i in range(len(accel_limits_list), len(lines))], sensor_names, loc='lower left', framealpha=0.8)
        ax.add_artist(legend2)
        
        # Add title and labels
        title = f' Weighted Acceleration (rms) values for the base curve ({self._dir} axis)\nbased on BS 6472'
        plt.title(title)
        plt.grid(True, which='both', linestyle='--', color=[0.1, 0.1, 0.1], alpha=0.1)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Acceleration rms (m/s²)', fontsize=11)
        plt.show()


    def plot_acceleration_limits(self, ax: plt.Axes, colormap: Colormap,
                                  act_fact: List[int], labels: List[str])-> Tuple[plt.Axes, List[List[float]]]:
        
        accel_limits_list = []
        for factor in act_fact:
            frequencies, accelerations = self.get_acc(factor)
            accel_limits_list.append(accelerations)


        for indx, (accelerations, label, lstyle) in enumerate(zip(accel_limits_list,labels,self.line_styles)):
            plt.plot(frequencies, accelerations, color= colormap.get_color(indx,len(accel_limits_list)),
                      linewidth=1.2, label= label, linestyle='--', dashes=lstyle) 

        return ax, accel_limits_list

    def plot_fft_rms(self, ax,data: np.array,rms_input: float, color='gray',tooltip = False, sensor_name = 'Normalized FFT' , weights = True):
        N = len(data)
        fft_output = np.fft.fft(data)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        magnitude = 2 * np.abs(fft_output) / N
        rms_per_bin = magnitude / np.sqrt(2)

        # Normalize rms_per_bin from 0 to 1
        normalized_rms = rms_per_bin / np.max(rms_per_bin)

        if weights & (self._dir == 'Z'):
            frequency, weighting_factors = self.get_weights()
            interp_weights = np.interp(np.abs(freqs), frequency, weighting_factors)
            weighted_rms = normalized_rms * interp_weights
        else:
            weighted_rms = normalized_rms

        # Multiply the input RMS with the normalized FFT
        adjusted_rms = weighted_rms * rms_input

        ax.loglog(freqs[:N//2], adjusted_rms[:N//2], color=color, label=sensor_name, alpha = 0.7)  # Added label here


        if tooltip:
            self.add_tooltip(ax,adjusted_rms=adjusted_rms, freqs=freqs, N=N)


        ax.grid(True, which='both', linestyle='--', color=[0.1, 0.1, 0.1], alpha=0.1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
    
    def add_tooltip(self,ax, adjusted_rms: np.array, freqs: np.array, N: int):
        # Detecting peaks
        peaks, _ = find_peaks(adjusted_rms[:N//2])
        
        # Find the maximum peak
        max_peak_index = peaks[np.argmax(adjusted_rms[peaks])]
        max_peak_freq = freqs[max_peak_index]
        max_peak_value = adjusted_rms[max_peak_index]

        # Define text position relative to peak
        text_x_offset = 0.5
        text_y_offset = 0.1

        # Check x boundaries and adjust
        if max_peak_freq + text_x_offset > ax.get_xlim()[1]:
            text_x_offset = -2

        # Check y boundaries and adjust
        if max_peak_value + text_y_offset > ax.get_ylim()[1]:
            text_y_offset = -0.5

        # Annotate the maximum peak
        ax.annotate(f'({max_peak_freq:.2f} Hz, {max_peak_value:.2f} m/s²)',
                    xy=(max_peak_freq, max_peak_value),
                    xytext=(max_peak_freq + text_x_offset, max_peak_value + text_y_offset),  
                    arrowprops=dict(facecolor='black', arrowstyle='->', shrinkA=0, shrinkB=5))

        if self.xlimits[1] < self.fs/2:
            ax.set_xlim(self.xlimits[0], self.xlimits[1])
        else: 
            ax.set_xlim(self.xlimits[0], int(self.fs/2))



