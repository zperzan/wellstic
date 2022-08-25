import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import lsq_linear


def convert_ectospc(ec, temp, tcc=0.021):
    """Given electrical conductivity and temperature, convert
    to specific conductance at 25 deg C.
    
    Parameters
    ----------
        ec : array of floats
            electrical conductivity
        temp : array of floats
            temperature in deg C
        tcc : float, default 0.021
            temperature compensation coefficient
    
    Returns
    -------
        spc : array of floats
            specific conductance at 25 deg C
    """
    
    spc = ec/(1+tcc*(temp - 25))

    return spc


def convert_to_ec(temp, spc):
    """Given a sequence of temperature values for an Oakton specific conductance
    reference solution, convert those values to electrical conductivity using the
    manufacturer's reference values. Oakton reports EC values for every degree or 5
    degrees; for intermediate temperature values, the EC value is interpolated linearly.
    
    Parameters
    ----------
        temp : array of floats
            the temperatures at which to calculate EC
        spc : {23, 84, 447, 1413, 8974, 15000}
            published specific conductivity of the calibration solution

    Returns
    -------
        ec : array of floats
            calculated electrical conductivity at each temperature
    """

    reftemp = np.asarray([5, 10, 15, 16, 17, 18, 19,
                          20, 21, 22, 23, 24, 25])

    # These are published specific conductivity values for Oakton conductivity
    # standards, though values may differ between batches and manufacturers
    ec_values = {23: np.asarray([15.32, 17.11, 18.32, 18.65, 18.97, 19.41, 19.85,
                                 20.3, 20.92, 21.51, 22.10, 22.55, 23.00]),
                 84: np.asarray([65, 67, 68, 70, 71, 73, 74,
                                 76, 78, 79, 81, 82, 84]),
                 447: np.asarray([278, 318, 361, 368, 376, 385, 394,
                                  402, 411, 419, 430, 438, 447]),
                 1413: np.asarray([896, 1020, 1147, 1173, 1199, 1225, 1251,
                                   1278, 1305, 1332, 1359, 1386, 1413]),
                 8974: np.asarray([5620, 6430, 7260, 7410, 7560, 7690, 7920,
                                   8090, 8280, 8450, 8590, 8800, 8974]),
                 15000: np.asarray([9430,  10720, 12050, 12280, 12590, 12900, 13190,
                                    13510, 13810, 14090, 14350, 14690, 15000])}

    # Interpolate onto tighter spacing
    x = np.linspace(reftemp.min(), reftemp.max(), 100000)
    ec_interp = np.interp(x, reftemp, ec_values[spc])

    ec = np.empty(temp.shape, dtype=float)

    for i, t in enumerate(temp):
        # Get idx of nearest temperature value
        idx = (np.abs(x - t)).argmin()
        ec[i] = ec_interp[idx]

    return ec


def filter_ringing(indf, threshold=-0.4):
    """Given a dataframe with "ringing" intensity values, filter out
    the faulty data.
    
    Parameters
    ----------
        df : dataframe
            a pandas dataframe containing "Intensity" data as one col 
        threshold : float, default is -0.4
            threshold fractional change used to determine whether a sudden 
            decrease from one record to the next is due to "ringing". Must 
            be between -1.0 and 0, exclusive
            
    Returns
    -------
        df : dataframe
            a copy of the original dataframe without ringing data
    """
    
    df = indf.copy()
    
    pct = df['Intensity'].pct_change() # note that this returns frac not pct
    mask = pct < threshold
    df = df[~mask]

    return df


def plot_intensitytemp(df, ccolor='steelblue', tcolor='firebrick'):
    """Plot a dataframe of intensity and temperature data
    
    Parameters
    ----------
        df : dataframe
            dataframe with DateTimeIndex containing 'Intensity' and 
            'Temp' columns
        ccolor : matplotlib color, default 'steelblue'
            color to use for plotting the intensity data
        tcolor : matplotlib color, default 'firebrick'
            color to use for plotting the temperature data
    
    Returns
    -------
        fig : matplotlib figure object
            figure handles
        ax : matplotlib axis object
            axes handles
    """
    fig, ax = plt.subplots(figsize=(8,5), tight_layout=True, facecolor='white')

    ax.plot(df.index, df['Intensity'], color=ccolor, label='Intensity')
    ax2 = ax.twinx()
    ax2.plot(df.index, df['Temp'], color=tcolor)
    
    # Add dummy plot for legend
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(df.index - pd.Timedelta("50W"), df['Temp'], color=tcolor, label='Temp')
    ax.set(ylabel='Intensity (lux)', xlim=xlim, ylim=ylim)
    ax2.set(ylabel=r'Temperature ($^\circ$C)')
    ax.legend()

    return fig, ax


def plot_dilution(df, column='SpC', label=None, ccolor='steelblue', 
                  tcolor='firebrick', figsize=(8,4)):
    """Given a dataframe and column name, plot the dilution curve.
    
    Parameters
    ----------
        df : dataframe
            dataframe that contains temperature data ('Temp' column) as well
            as the dilution variable of interest, with DateTimeIndex
        column : string
            column containing the dilution variable of interest
        label : tuple of string
            tuple containing (legend label, y-axis label) for the plot
        ccolor : matplotlib color, default 'steelblue'
            plotted color of the dilution data
        tcolor : matplotlib color, default 'firebrick'
            plotted color of the temperature data
        figsize : tuple of float, default (8,4)
            size of the matplotlib figure
    
    Returns
    -------
        fig : matplotlib figure object
            figure handles
        ax : matplotlib axis object
            axes handles
        
    """
    
    # Determine labels
    if label is None:
        if column == 'SpC':
            leglabel = 'Spec. cond.'
            axlabel = 'Specific conductance ($\mu$S/cm)'
        elif column == 'EC':
            leglabel = 'Conductivity'
            axlabel = 'Electrical conductivity ($\mu$S/cm)'
        elif column == 'Intensity':
            leglabel = 'Intensity'
            axlabel = 'Intensity (lux)'
    elif isinstance(label, list):
        leglabel = label[0]
        axlabel = label[1]
    
    fig, ax = plt.subplots(figsize=figsize, tight_layout=True, facecolor='white')

    ax.plot(df.index, df[column], color=ccolor, label=leglabel)
    ax2 = ax.twinx()
    ax2.plot(df.index, df['Temp'], color=tcolor)
    
    # Add dummy plot for legend
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(df.index - pd.Timedelta("50W"), df['Temp'], color=tcolor, label='Temp')
    ax.set(ylabel=axlabel, xlim=xlim, ylim=ylim)
    ax2.set(ylabel='Temperature ($^\circ$C)')
    ax.legend()

    return fig, ax


def transform(X):
    """Transform bivariate X to a 2nd order bivariate polynomial
    without x1*x2 terms. In other words, transform [x1, x2] to 
    [1, x1, x2, x1**2, x2**2]
    
    Parameters
    ----------
        X : array of floats
            bivariate data to transform, of dimension (n, 2)
    
    Returns
    -------
        X_t : array of floats
            transformed data, of dimension (n, 5)
    """
    
    # Generate a model of polynomial features
    poly = PolynomialFeatures(degree=2)

    # Transform X 
    # Note that this generates [1, x1, x2, x1**2, x1*x2, x2**2]
    X_t = poly.fit_transform(X)

    # Delete cross term x1*x2
    X_t = np.delete(X_t,(4),axis=1)

    # Rearrange so the columns are in the same order as in Gilman
    X_t = X_t[:, [0, 2, 1, 4, 3]]
    
    return X_t


def fit(X, y):
    """Fit Gillman polynomial to X, y calibration data using a constrained
    least squares regression.
    
    Parameters
    ----------
        X : array of float
            bivariate X data of dimension (n, 2)
        y : array of float
            univariate y data
    
    Returns
    -------
        result : dict
            least squares results with the following keys:
            
            'x' : array of floats
                coefficients for the least squares solution
            'cost' : float
                value of the cost function
            'fun' : array of floats
                vector of residuals for the solution
            'nit' : int
                number of iterations
            'status' : int
                reason the search stopped. status=3 if the unconstrained
                solution is optimal
            'message' : str
                verbal description of 'status'
            'success' : bool
                whether or not the solver converged
    """
    
    X_t = transform(X)
    llim = np.array([-np.inf, 0, -np.inf, 0, -np.inf])
    ulim = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])
    result = lsq_linear(X_t, y, bounds=(llim, ulim), lsmr_tol='auto')
    
    return result


def predict(X, coefs):
    """Given lux/temperature data, convert it to EC using the 
    given polynomial coefficients.
    
    Parameters
    ----------
        X : 2d array of floats
            Array of shape n x 2 containing intensity (lux) on the 0th axis
            and temperature on the 1st axis
        coefs : tuple of floats
            Gillman coefficients a, b, c, d, e
            
    Returns
    -------
        """
    
    X_t = transform(X)
    y_hat = np.matmul(coefs, X_t.T)
    
    return y_hat


def fit_predict(X, y, X1):
    """Fit Gillman polynomial to Intensity/temp (X), and EC (y) 
    and predict EC on X1.
    
    Parameters
    ----------
        X : array of floats
            bivariate X data of dimension (n, 2)
        y : array of floats
            univariate y data
        X1 : array of floats
            bivariate X data on which to predict y_hat

    Returns
    -------
        y_hat : array of floats
            predicted EC for X1
    """
    
    result = fit(X, y)
    coefs = result['x']

    y_hat = predict(X1, coefs)
 
    return y_hat


def int_error(I):
    """Estimate measurement error on the measured intensity, using parameters from 
    a 2nd-order polynomial fit.
    
    Parameters
    ----------
        I : array
            1-d array of shape (n,) of measurement intensity
    
    Returns 
    -------
        I_err : array
            1-d array of shape (n,) of the estimated intensity error 
    """

    # Load fit parameters (from sandbox.ipynb)
    A = np.load('src/IntensityErrorParams.npy')
    
    I_trans = np.ones((I.shape[0], 3), dtype=float)
    I_trans[:, 1] = I
    I_trans[:, 2] = I**2
    I_err = np.squeeze(A @ I_trans.T)

    return I_err


def fit_statistics(X, y):
    """Calculate adjusted rsqaured, residual error and the covariance 
    matrix of the fit coefficients

    Parameters
    ----------
        X : array of floats
            bivariate X data of dimension (n, 2)
        y : array of floats
            univariate y data

    Returns
    -------
        adjusted_rsq : float
            adjusted rsquared
        residual_error : float
            residual error
        cov : array of floats
            covariance matrix of the fit coefficients
    """
    
    fi = fit_predict(X, y, X)
    
    n = y.shape[0]
    X_t = transform(X)
    p = X_t.shape[1]
    
    # Calculate residual sum of squares
    res = fi - y
    ssres = np.sum(res**2)

    # Calculate total sum of squares
    sstot = np.sum((y - np.mean(y))**2)
    rsquared = 1 - (ssres/sstot)

    # Calculate adjusted r-squared
    n = y.shape[0]
    p = X.shape[1]
    adjusted_rsq = 1 - ((n - 1)/(n - 1 - p))*(1-rsquared)

    # Calculate residual standard error
    sr = np.sqrt(ssres/(n-2))
    
    # Calculate covariance of fit coefficients
    varres = np.var(res)
    covb = varres * np.linalg.inv(X_t.T @ X_t)

    return adjusted_rsq, sr, covb


def prop_error(T, I, covb, coefs, incl_cov=False):
    """Calculate standard deviation of predicted EC using error propagation on 
    the fit regression coefficients and error on intensity. Note that this does 
    not include error in measurement of temperature.
    
    Parameters
    ----------
        T : array
            1-d array of shape (n,) of measurement temperature
        I : array
            1-d array of shape (n,) of measurement intensity
        covb : array
            covariance matrix of the regression coefficients
        coefs : array
            coefficients used in the calibration curve
        incl_cov : bool
            whether or not to include covariance terms in error propagation (note that 
            doing so often causes totatl error to decrease)
    
    Returns 
    -------
        ec_std : array
            1-d array of shape (n,) of the standard deviation of each predicted EC
    """

    # Calculate uncertainty on the measured intensity
    _, b, _, d, _ = coefs
    I_err = int_error(I)
    I_term = (b + 2*d)*I_err
    
    # Now calculate error terms from uncertainty in each regression coefficient
    a_term = covb[0, 0]
    b_term = I**2 * covb[1, 1]
    c_term = T**2 * covb[2, 2]
    d_term = I**4 * covb[3, 3]
    e_term = T**4 * covb[4, 4]

    ab_term = I*covb[0, 1]
    ac_term = T*covb[0, 2]
    ad_term = I**2 * covb[0, 3]
    ae_term = T**2 * covb[0, 4]
    bc_term = I*T*covb[1, 2]
    bd_term = I**3 * covb[1, 3]
    be_term = I*T**2 * covb[1, 4]
    cd_term = T*I**2 * covb[2, 3]
    ce_term = T**3 * covb[2, 4]
    de_term = I**2 * T**2 * covb[3, 4]

    ec_var = I_term + a_term + b_term + c_term + d_term + e_term 
    
    # Whether or not to include coviarance terms
    # This often will decrease total error because of negative covariance
    if incl_cov:
        ec_var += 2*ab_term + 2*ac_term + 2*ad_term + 2*ae_term + \
                  2*bc_term + 2*bd_term + 2*be_term + 2*cd_term + 2*ce_term + 2*de_term 

    ec_std = np.sqrt(ec_var)
    
    return ec_std
