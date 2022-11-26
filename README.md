# WellSTIC Calibration

This repository contains code and examples for fitting a calibration curve for a WellSTIC specific conductance sensor. For more information see: 

Perzan, Z. & Chapin, T. (2022). WellSTIC: A cost-effective sensor for performing point dilution tests to measure groundwater velocity in shallow aquifers. 

The jupyter notebook `jupyter/example_calibration.ipynb` walks through the example application and contains markdown cells that describe each step in the process. The `calibration data` folder contains example .csv files of recorded data during the calibration process. Running the example juptyer notebook fits a calibration curve to these data and saves the fit calibration coefficients to `calibration_coefficients`. Within this directory, `WellSTIC1_CalibrationCoefficients.npy` contains the *a*, *b*, *c*, *d* and *e* coefficients corresponding to equation 2 in Perzan and Chapin (2022). The output file `calibration_coefficients/WellSTIC1_covb.npy` contains the covariance matrix for these fit coefficients, which can be used to propagate calibration uncertainty through to computed specific conductance values. A WellSTIC python module can be found at `jupyter/src/wellstic.py`.


## Python dependencies
- `numpy`
- `matplotlib`
- `scipy`
- `pandas`
- `sklearn`
