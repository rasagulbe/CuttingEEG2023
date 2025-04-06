Code from the workshop given during the [CuttingGardens 2023](https://cuttinggardens2023.org/gardens/frankfurt/) conference (Frankfurt Garden).

### [Extracting interpretable patterns from multivariate time series data using GED](https://osf.io/b7zgf/wiki/home/)

Electrophysiological signals recorded at any spatial scale represent a mix of activity from different neural sources. This tutorial focuses on a supervised, as opposed to blind, source separation using generalized eigenvalue decomposition (GED). GED method extracts statistical (rather than anatomical) sources, which are expressed as linear weighted combinations of electrode time series (spatial filters). Supervision step involves data selection for implementing GED, as the method relies on the covariance matrices derived from different features of the data (e.g. narrow-band vs. broad-band activity; pre-stimulus vs. post-stimulus period; two experimental conditions). To illustrate the strengths (increased signal-to-noise ratio, dimensionality reduction) and limitations of the method, several empirical EEG datasets are used, including visual steady-state evoked potentials and a cognitive control task. Practical aspects are emphasized, such as the effects of noise, the amount of data used, covariance matrices derived from single versus concatenated trials, matrix regularization, overfitting, and interpretation of GED results.
##
![Github_CuttinEEG2023](https://github.com/user-attachments/assets/e10c1ae6-c12c-4074-bb90-dc241f766f4b)
##
### Repository contains MATLAB code to run the analysis using a sample dataset.

An auxiliary_code folder contains additional scripts from external sources required for running the analysis.
##
### Dataset 
Sample dataset can by found on OSF repository (https://osf.io/b7zgf/).
##
### Requirements
EEGLAB toolbox is needed for plotting scalp topographies using `topoplot` function. BrewerMap is used for ColorBrewer 2.0 colorschemes in MATLAB.


