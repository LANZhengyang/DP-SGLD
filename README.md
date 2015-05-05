# DP-SGLD
Trying to replicate the results of Figure 2 in this paper: http://arxiv.org/pdf/1502.07645.pdf

Stochastic Gradient Langevin Dynamics (SGLD) and the DP version of it for logistic regression were implemented. A 
comparison of the prediction results (loglikelihood and prediction accuracy for test data) are displayed in the two 
plots. However, what I got so far is not as expected: the prediction should be more accurate as log epsilon increases, 
and finally it should get close to the Non DP version results. But my replicate doesn't show this pattern, especially 
for the loglikelihood results.
