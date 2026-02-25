# can call matlab functions from python. do that. requires matlab licence in runtime.
import matlab.engine
from pathlib import Path


workspace_root = Path(__file__).resolve().parents[1]
b2bdc_folder = workspace_root / "libs" / "B2BDC_v1.0"
b2bdc_project_folder = workspace_root / "B2BDC"

eng = matlab.engine.start_matlab()

if b2bdc_folder.exists() and b2bdc_project_folder.exists():
    eng.addpath(str(b2bdc_folder))
    eng.addpath(eng.genpath(str(b2bdc_folder)))
    eng.addpath(str(b2bdc_project_folder))
    print(f"Success: Added {b2bdc_folder} and {b2bdc_project_folder} to MATLAB path.")
else:
    raise FileNotFoundError(
        f"Required folders are missing. B2BDC: {b2bdc_folder.exists()}, project B2BDC: {b2bdc_project_folder.exists()}"
    )

result = eng.consistency(nargout=1)
print("B2BDC consistency run finished.")
print(result)

eng.quit()