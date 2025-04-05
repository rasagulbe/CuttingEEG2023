# [CuttingEEG 2023](https://cuttinggardens2023.org/gardens/frankfurt/)
Code from the workshop given during the CuttingGardens 2023 conference (Frankfurt Garden).

## Extracting interpretable patterns from multivariate time series data using GED

Electrophysiological signals recorded at any spatial scale represent a mix of activity from different neural sources. This tutorial focuses on a supervised, as opposed to blind, source separation using generalized eigenvalue decomposition (GED). GED method extracts statistical (rather than anatomical) sources, which are expressed as linear weighted combinations of electrode time series (spatial filters). Supervision step involves data selection for implementing GED, as the method relies on the covariance matrices derived from different features of the data (e.g. narrow-band vs. broad-band activity; pre-stimulus vs. post-stimulus period; two experimental conditions). To illustrate the strengths (increased signal-to-noise ratio, dimensionality reduction) and limitations of the method, several empirical EEG datasets are used, including visual steady-state evoked potentials and a cognitive control task. Practical aspects are emphasized, such as the effects of noise, the amount of data used, covariance matrices derived from single versus concatenated trials, matrix regularization, overfitting, and interpretation of GED results.
##

Repository contains MATLAB code to run the analysis using sample dataset (can be found here:)
An auxiliary_code folder contains additional scripts required for running the analysis. These scripts were adapted from external sources.

### Requirements
EEGLAB toolbox is needed for plotting scalp topographies using `topoplot` function.


