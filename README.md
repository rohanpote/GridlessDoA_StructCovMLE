# Maximum likelihood-based gridless Direction of Arrival (DoA) estimation

The code implements the algorithms proposed in the following paper, and also generates the plots presented in the same:

R. R. Pote and B. D. Rao, "Maximum Likelihood-Based Gridless DoA Estimation Using Structured Covariance Matrix Recovery and SBL With Grid Refinement," in IEEE Transactions on Signal Processing, vol. 71, pp. 802-815, 2023

Please refer the above paper if you find the proposed ideas or code relevant for your work.

# Description of the work: 
For the case when measurements are spatial (or temporal) samples at regular intervals, Sparse Bayesian Learning (SBL) objective is reparameterized, leading to a structured matrix recovery technique based on maximum likelihood estimation. A majorization-minimization based iterative procedure is implemented to estimate the structured matrix; each iteration solves a semidefinite program.  The DoAs are recovered in a gridless manner using decomposition of positive semidefinite Toeplitz matrices. 
For the general case of irregularly spaced samples, an iterative SBL procedure is implemented that refines grid points to increase resolution near potential source locations, while maintaining a low per iteration complexity.
The proposed correlation-aware approaches is more robust to issues such as fewer snapshots, correlated or closely separated sources, and improves sources identifiability.
