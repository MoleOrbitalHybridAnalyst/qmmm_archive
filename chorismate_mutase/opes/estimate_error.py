import numpy as np
from statsmodels.tsa import ar_model
from sys import argv

def sample_variance(noise_sigma, rho, n):
    var = noise_sigma**2 / (1 - rho**2)
    var /= (n**2)
    var *= (n + 2 * n * rho / (1 - rho) + 2 * (rho**n - 1) / (1 - rho)**2 * rho)
    return var

def sample_exp_variance(noise_sigma, rho, n, delta):
    lnx_mu = delta / (1 - rho)
    lnx_s2 = noise_sigma**2  / (1 - rho**2)
    x_mu = np.exp(lnx_mu + lnx_s2 / 2)
    expsum = 0
    for i in range(n):
        for j in range(n):
            expsum += np.exp(lnx_s2 * rho**abs(i-j))
    expsum /= n**2
    return x_mu**2 * (expsum - 1)

def print_res(ts, res):
    delta_, rho_ = res.params
    lnx_mu = delta_ / (1 - rho_)
    lnx_s2 = res.sigma2 / (1 - rho_**2)
    print("estimated lnx mean:", lnx_mu)
    print("sample lnx mean:", np.mean(ts))
    print("estimated rho:", rho_)
    print("estimated lnx variance:", lnx_s2)
    lnx_sample_var = sample_variance(np.sqrt(res.sigma2), rho_, len(ts))
    print("estimated lnx sample variance:", lnx_sample_var)

    x_mu = np.exp(lnx_mu + lnx_s2 / 2)
    print("estimated x mean:", x_mu)
    print("sample x mean:", np.mean(np.exp(ts)))
    x_sample_var = sample_exp_variance(np.sqrt(res.sigma2), rho_, len(ts), delta_)
    print("estimated x sample variance:", x_sample_var)
    return x_sample_var

if __name__ == '__main__':
    ts = np.loadtxt(argv[1])[:,2]
    N = len(ts)

    # assume ts is arriving time that obeys log-normal distr.
    # changes it into noraml distribution
    ts = np.log(1 / ts)

    # try to minimize sample variance by removing some initial data
    minvar = np.inf
    min_t = None
    stepsize = 1
    for i in range(max(1, N // stepsize // 2)):
        m = ar_model.AutoReg(ts[i*stepsize:], 1, seasonal=False)
        res = m.fit()
        print("----------------------------------------------")
        print(f"using data[{i*stepsize}:]")
        print("----------------------------------------------")
        svar = print_res(ts[i*stepsize:], res)
        print("")
        
        if svar < minvar:
            minvar = svar
            min_t = i

    print("----------------------------------------------")
    print(f"min var found with data[{min_t*stepsize}:]")
    print("----------------------------------------------")
    ts_ = ts[min_t*stepsize:]
    m = ar_model.AutoReg(ts_, 1, seasonal=False)
    res = m.fit()
    var = print_res(ts[min_t*stepsize:], res)
    print(f"final result: {np.mean(np.exp(ts_))} +- {np.sqrt(var)}")
    print("")

