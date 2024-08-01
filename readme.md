## BS 6472

Serviceability assessment as per BS 6472 for human exposure to vibration in buildings Vibration sources other than blasting

![Result][data/grandstand_N02_force/example_bs6473.svg]
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

    acc_data = vibration_survey.list_of_tests[5].Acc_z
    fs = vibration_survey.list_of_tests[5].fs
    _dir = 'Z'
    activity_factor = 24
    rms = 0.5

    service_assessment = Service_assessment(acc_data, fs, _dir, activity_factor,rms)
    service_assessment.BS_6472()
```

# Install
```sh
pip install -r requirements.txt
```
