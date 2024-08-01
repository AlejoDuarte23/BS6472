## BS 6472

Serviceability assessment as per BS 6472 for human exposure to vibration in buildings Vibration sources other than blasting

![Result](data/example_bs6473.svg)
## How to use it 

Import  the Service_assessment class and pass the accelerarion data (NDArray), sampling frequency (float), activaty factor based on BS6472 guidelines, and rms

# Example

```python 
    
    from bs_6473 import Service_assessment

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

    # Process the data: and slice the evet : [39600:42000]
    acc_data = vibration_survey.list_of_tests[5].Acc_z[39600:42000]*9.81
    fs = vibration_survey.list_of_tests[5].fs
    _dir = 'Z'
    service_assessment = Service_assessment(acc_data, fs, _dir,0.18)
    
    # curve factor as per BS 6472
    curve_factors = [1,2,4,8,24]
    labels = [f"{factor}x base curve" for factor in curve_factors]
    labels[0] = 'Base curve'
    
    # Assessment plot
    service_assessment.BS_6472(act_fact=[1,2,4,8,24], labels=labels)
```

# Install
```sh
pip install -r requirements.txt
```