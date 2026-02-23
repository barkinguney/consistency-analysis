# can call matlab functions from python. do that. requires matlab licence in runtime.
import matlab.engine
from pathlib import Path


b2bdc_folder = Path("libs/B2BDC_v1.0")

eng = matlab.engine.start_matlab()

if b2bdc_folder.exists():
    eng.addpath(str(b2bdc_folder))
    eng.addpath(eng.genpath(str(b2bdc_folder)))
    print(f"Success: Added {b2bdc_folder} to MATLAB path.")
else:
    print(f"Error: Could not find {b2bdc_folder}")
    
# file = b2bdc_folder / "demo" / "vectorConsistencyDemo.m"
# demo_result = eng.vectorConsistencyDemo(nargout=0)

#file = b2bdc_folder / "demo" / "GRIMech_demo2.m"
demo_result = eng.GRIMech_demo2(nargout=0)

print(eng.sqrt(100.0))
eng.quit()