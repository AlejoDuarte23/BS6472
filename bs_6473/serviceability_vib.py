
import matplotlib.pyplot as plt
import numpy as np
import sys 
import json 


from pathlib import Path
from scipy.signal import find_peaks
from pathlib import Path
from typing import TypeVar, Union, List, overload
from matplotlib.colors import LinearSegmentedColormap

plt.close('all')
T = TypeVar('T',bound= Union[int, List] )




class Service_assessment():
    def __init__(self,acc_data,fs,_dir,rms_fix = None):
        self.data = acc_data 
        self.fs = fs
        self._dir = _dir 
        self.x_ticks = [1, 2, 3, 5, 6.3, 10, 16, 25, 40, 63, 80]
        self.ylimits = []
        self.xlimits = []
        self.rms_fix = rms_fix
        self.line_styles = [
                (5, 5), (10, 5), (15, 5), (20, 5), (25, 5),
                (5, 10), (5, 15), (5, 20), (5, 25), (10, 10)
            ]

    
  
    def get_acc(self, act_fact: int):
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
                tooltip: list,
                sensor_names: list):
    
        self.tooltip = tooltip
        self.sensor_names = sensor_names
        colormap = Colormap()


        if isinstance(act_fact, int):
            act_fact = [act_fact]

        if isinstance(labels, str):
            labels = [labels]

        accel_limits_list = []
        title = f'Acceleration (rms) values for the base curve ({self._dir} axis)\nbased on BS 6472'

        for factor in act_fact:
            frequencies, accelerations = self.get_acc(factor)
            accel_limits_list.append(accelerations)

        fig, ax = plt.subplots()
        for accelerations, label, lstyle in zip(accel_limits_list,labels,self.line_styles):
            plt.plot(frequencies, accelerations, color='r', linewidth=1.2, label= label, linestyle='--', dashes=lstyle)  # Added label here

        # Create the first legend

        # Add the legend to the plot'+}        
        if isinstance(self.rms_fix, float):
            self.data = [self.data]
            self.rms_fix = [self.rms_fix]
            self.tooltip = [True]
            self.sensor_names = [sensor_names]
        
        number_of_signals = len(self.data)
        for index, (signal, rms, tooltip_val,sensor_name) in enumerate(zip(self.data, self.rms_fix, self.tooltip , self.sensor_names)):    
            self.plot_fft_rms_fix(ax, signal, rms, color=colormap.get_color(index, number_of_signals),tooltip=tooltip_val,sensor_name = sensor_name)


        plt.title(title)
        plt.grid(True, which='both', linestyle='--', color=[0.1, 0.1, 0.1], alpha=0.1)
        ax.set_xlabel('Frequency (Hz)')
        plt.ylabel('Acceleration rms (m/s²)', fontsize=11)
        plt.legend(framealpha=0.0, loc='lower left')
        ax.legend(loc='lower left')
        plt.show()
    def set_rms(self,rms):
        self.rms_fix = rms
        

    
    def plot_fft_rms_auto(self, ax, color='gray'):
        N = len(self.data)
        fft_output = np.fft.fft(self.data)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        magnitude = 2 * np.abs(fft_output) / N
        rms_per_bin = magnitude / np.sqrt(2)
        
        ax.semilogx(freqs[:N//2], rms_per_bin[:N//2], color='green', label='Normalized FFT')  # Added label here
    
        # Detecting peaks
        peaks, _ = find_peaks(rms_per_bin[:N//2])
        
        # Find the maximum peak
        max_peak_index = peaks[np.argmax(rms_per_bin[peaks])]
        max_peak_freq = freqs[max_peak_index]
        max_peak_value = rms_per_bin[max_peak_index]
        
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
        
        ax.grid(True, which='both', linestyle='--', color=[0.1, 0.1, 0.1], alpha=0.1) # Use ax.grid


    def plot_fft_rms_fix(self, ax,data: np.array,rms_input: float, color='gray',tooltip = False, sensor_name = 'Normalized FFT'):
        N = len(data)
        fft_output = np.fft.fft(data)
        freqs = np.fft.fftfreq(N, 1/self.fs)
        magnitude = 2 * np.abs(fft_output) / N
        rms_per_bin = magnitude / np.sqrt(2)

        # Normalize rms_per_bin from 0 to 1
        normalized_rms = rms_per_bin / np.max(rms_per_bin)

        # Multiply the input RMS with the normalized FFT
        adjusted_rms = normalized_rms * rms_input

        ax.loglog(freqs[:N//2], adjusted_rms[:N//2], color=color, label=sensor_name, alpha = 0.8)  # Added label here


        if tooltip:
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

        ax.grid(True, which='both', linestyle='--', color=[0.1, 0.1, 0.1], alpha=0.1)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')


class Colormap:
    def __init__(self):
        colors = ["#0F0", "#00F"]  # Dark grey to Gold
        n_bins = 7  # Number of bins
        cmap_name = "grey_gold"
        self.cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    def get_color(self, index, total):
        """Get a specific color from the colormap based on index and total number of items."""
        return self.cm(index / (total - 1) if total > 1 else 0)
