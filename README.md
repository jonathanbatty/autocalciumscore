### BSc(Hons) Computer Science
### CM3070 Final Project
### A 3-dimensional (3D), regression-based, deep learning approach to automate quantification of coronary artery calcification using thoracic computed tomography (CT) images                #

#### Submitted in partial fulfilment for BSc Computer Science | University of London | 02/09/2022
#### Jonathan Batty | UoL Student Number: 190164315/1

This project requires Python 3.

Installation instructions to run the Jupyter Notebook and CLI tool locally:
1. Download this repository to a local folder (note: this repo does not contain the ~19.5 GB of CT data used to train the model)
1. Create a new virtual environment using: `python -m venv venv` or an equivalent command.
1. Activate the virual environment, using: `venv\Scripts\activate.bat`.
1. Install the libararies required by this project, by running: `pip install -r requirements.txt`.
1. Open the Jupyter Notebook by starting a new JupyterLab instance, by running `jupyter lab`.
1. Run the command line prediction tool by running: `python coronary-artery-scoring.py scans_to_analyse.txt output'. where `scans_to_analyse` is a text file containing a DICOM directory on each line and `output` specifies the name of the output file.
