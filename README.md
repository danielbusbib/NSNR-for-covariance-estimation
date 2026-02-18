# NSNR-for-covariance-estimation
code of the journal paper: Normalized Signal-to-Noise Ratio for Covariance Estimation in Target Detection

## Abstract


## KL BASED LOOCV SHRINKAGE ESTIMATOR
```python
def loocv_loglike(X, alpha):
    D, N = X.shape
    S = X @ X.T / N
    T = np.trace(S)/D * np.eye(D)
    C = (1 - alpha) * S + alpha * T
    invC = np.linalg.inv(C)
    z = np.einsum('ji,jk,ki->i', X, invC, X)
    trace_term = np.mean(z / (1 - (1 - alpha) / N * z))
    log_det = np.log(np.linalg.det(C))
    dist = trace_term + log_det
    return dist


def loocv(X):
  alphas = np.logspace(-3,-.01,20)
  B, M, D = X.shape
  Chat = np.zeros((B, D, D))
  for b in range(B):
    Xi = X[b,:,:].T
    distances = np.array([loocv_loglike(Xi, alpha) for alpha in alphas])
    optimal_alpha = alphas[np.argmin(distances)]
    S = Xi @ Xi.T / M
    Chat[b] = (1-optimal_alpha) * S + optimal_alpha * np.trace(S)/D * np.eye(D)
  return Chat

```

## Citation

Please cite if you are using this code for your research:

```

coming soon

```
