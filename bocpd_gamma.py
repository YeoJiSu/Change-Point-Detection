"""============================================================================
Author: Gregory Gundersen

Python implementation of Bayesian online changepoint detection for a normal
model with unknown mean parameter. For algorithm details, see
https://github.com/gwgundersen/bocd/blob/master/bocd.py

    Adams & MacKay 2007
    "Bayesian Online Changepoint Detection"
    https://arxiv.org/abs/0710.3742

For Bayesian inference details about the Gaussian, see:

    Murphy 2007
    "Conjugate Bayesian analysis of the Gaussian distribution"
    https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

This code is associated with the following blog posts:

    http://gregorygundersen.com/blog/2019/08/13/bocd/
    http://gregorygundersen.com/blog/2020/10/20/implementing-bocd/
    
============================================================================"""

import matplotlib.pyplot as plt
from   matplotlib.colors import LogNorm
import numpy as np
from   scipy.stats import norm
from   scipy.special import logsumexp
import pandas as pd
from scipy.stats import gamma

# -----------------------------------------------------------------------------

def bocd(data, model, hazard):
    """Return run length posterior using the BOCD algorithm."""
    T = len(data)
    log_R = -np.inf * np.ones((T+1, T+1))
    log_R[0, 0] = 0  # log 1 == 0
    log_message = np.array([0])  # log 1 == 0
    log_H = np.log(hazard)
    log_1mH = np.log(1 - hazard)

    for t in range(1, T+1):
        x = data[t-1]
        if x == 0:
            x = x + 0.00001

        # Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)
        
        # Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)
        
        # Determine run length distribution.
        log_R[t, :t+1] = new_log_joint
        log_R[t, :t+1] -= logsumexp(new_log_joint)

        # Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R

# -----------------------------------------------------------------------------


class GammaModel:
    def __init__(self, alpha0, beta0):
        """Initialize model with Gamma distribution parameters."""
        self.alpha0 = alpha0
        self.beta0 = beta0
        self.alpha_params = np.array([alpha0])
        self.beta_params = np.array([beta0])

    def log_pred_prob(self, t, x):
        """Compute log of the predictive probabilities for each run length hypothesis."""
        alpha = self.alpha_params[:t]
        beta = self.beta_params[:t]
        # print(alpha, beta)
        return gamma.logpdf(x, a=alpha, scale=1/beta)

    def update_params(self, t, x):
        """Update the parameters of the Gamma distribution upon observing a new datum."""
        new_alpha_params = self.alpha_params + 0.3  # Update alpha by adding 1 for each observation
        new_beta_params = self.beta_params + x   # Update beta by adding the new data point
        self.alpha_params = np.append([self.alpha0], new_alpha_params)
        self.beta_params = np.append([self.beta0], new_beta_params)
# -----------------------------------------------------------------------------

def calculate_returns(data):
    """Calculate returns from price data."""
    returns = 1 + np.diff(data) / data[:-1]
    # returns = np.log(1 + returns)  # log(1 + return) to handle negative and zero returns
    return np.append([1], returns)  # Append a 0 for the first return to maintain length


# -----------------------------------------------------------------------------

def plot_posterior(T, data, returns, cps, R, cp_10):
    fig, axes = plt.subplots(3, 1, figsize=(20,10))

    ax1, ax3, ax2 = axes

    ax1.scatter(range(0, T), data, s =5)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    
    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r', 
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)

    for cp in cps:
        ax1.axvline(cp, c='red', ls='dotted', label='Actual CP')
        ax3.axvline(cp, c='red', ls='dotted', label='Actual CP')
     
    for cp in cp_10:
        ax2.axvline(cp, c='purple', ls='dotted', lw=1)
        # ax2.scatter(cp, 50, color='brown', marker='o', s=2, label = 'Predicted CP (r_t = 20)')
    # 범례 추가
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 중복 라벨 제거
    # ax1.legend(by_label.values(), by_label.keys())
    ax2.legend(by_label.values(), by_label.keys(), loc = 'center right')
    ax3.scatter(range(0, T), returns, s=5)
    ax3.plot(range(0, T), returns)
    ax3.set_xlim([0, T])
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------

if __name__ == '__main__':
    file_path = 'train_apple.csv'
    df = pd.read_csv(file_path)
    data = df['Close'].values

    # # Calculate returns
    # returns = calculate_returns(data)
    
    # Min-Max Scaling 수행
    returns = (data - data.min()) / (data.max() - data.min())

    
    # returns = data
    print(np.mean(returns)) # 1.00 = α/β
    
    T = len(returns)
    
    beta0 = 1 # 데이터의 평균이 1 근처라고 가정할 경우, α/β=1이 되도록 설정할 수 있습니다.
    alpha0 = beta0 * np.mean(returns)
    
    hazard = 1/250

    model = GammaModel(alpha0, beta0)
    R = bocd(returns, model, hazard)
    
    
    # 'label' 열에서 값이 1인 인덱스를 추출합니다.
    cps = df[df['label'] != 0].index.tolist()
    print("real cp", cps)
    
    # CP 출력하기 
    cp = []
    cp_1 = []  # cp 배열 초기화
    cp_2 = []
    cp_3 = []
    cp_10 = []
    
    # 각 행에 대해서 검사
    for t in range(T+1):
        # R[t, 0]가 t행의 최대값인지 확인
        if R[t, 0] == np.max(R[t, :]):
            cp.append(t)
        if R[t, 1] == np.max(R[t, :]):
            cp_1.append(t-1)
        if R[t, 2] == np.max(R[t, :]):
            cp_2.append(t-2)
        if R[t, 3] == np.max(R[t, :]):
            cp_3.append(t-3)
        if R[t, 20] == np.max(R[t, :]):
            cp_10.append(t-20)
            

    # 최종 cp 배열 출력
    print("pred_cp",cp)
    print("pred_cp_1", cp_1)
    print("pred_cp_2", cp_2)
    print("pred_cp_3", cp_3)
    print("pred_cp_10", cp_10)
    
    # plt.plot(R[1:,0])
    plot_posterior(T, data,returns, cps, R, cp_10)
    
    
    