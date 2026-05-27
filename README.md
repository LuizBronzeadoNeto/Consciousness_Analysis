### Abstract
This is the source code for results described in the paper Classification of States of Consciousness in Individuals under GABAergic Anesthesia Using Machine Learning Algorithms, it contains:
- analysis.py
- confidence_interval_auc.ipynb
- statistical_analysis.py
### dependencies
This code makes use of several data processing, machine learning and optimization libraries, to install them using pip:
```
pip install pandas
pip install numpy
pip install numba
pip install -U scikit-learn
```
In Linux, dependencies can also be installed globally using your distribution's package manager (for Arch based distros, numba can be found on the AUR).
### Analysis
The analysis.py file contains all code related to data processing, model training and validation, it makes use of the functions in complexity_calculations.py to compute complexity and entropy metrics.

Usage:
```
python analysis.py <GPU_arg>
```
Run GPU argument as either true or false to make use of GPU acceleration
### Statistical analysis
Contains code that checks the statistical significance of each individual feature used in classification.

Usage:
```
python statistical_analysis.py
```
### AUC confidence interval
A notebook containing a deeper analyisis with a 95% CI for various slices of the data.

It can be run using a traditional jupyter notebook environment:
```
jupyter-notebook confidence_interval_auc.ipynb
```
### Sources
Abel, J., Badgeley, M., Meschede-Krasa, B., Schamberg, G., Garwood, I., Lecamwasam, K., Chakravarty, S., Zhou, D., Keating, M., Purdon, P., & Brown, E. (2021). Multitaper spectra recorded during GABAergic anesthetic unconsciousness (version 1.0.0). PhysioNet. RRID:SCR_007345. [[https://doi.org/10.13026/m792-h077]]
