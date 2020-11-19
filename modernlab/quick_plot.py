import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import inspect as inspect

# How to use
# Place in your x data and y data, then put tites in as strings
# you can use the arguments positionally or just call them by name
# y_bar indicates if you should plot error bars

def quick_plot(xdata, ydata, xname = None, yname = None, title = None, fitname = None, dataname = None, yerror = None, xerror = None, fit = None, y_bar = False, x_bar = False, legend = False, guesses = None, return_raw = False):
    '''
    Plots data using matplotlib.pyplot with the ability to add tiles, a fit line, error bars, and more. Used to plot basic plots quickly. Please refer to the notes for best practices for details on how to use the function properly.

        Parameters:
                xdata (array_like): x position of data points
                ydata (array_like): y position of data points
                xname (str, optional): name of x axis 
                yname (str, optional): name of y axis
                title (str, optional): plot title
                fitname (str, optional): name shown in legend for the fit function
                dataname (str, optional): name shown in legend for the graphed data points
                yerror (array_like): error for each value in ydata 
                xerror (array_like): error for each value in xdata
                fit (Callable, optional): function to fit to the xdata and ydata 
                    see f parameter in scipy.optimize.curve_fit
                y_bar (bool, optional): determines if vertical error bars are graphed with the xdata and ydata
                x_bar (bool, optional): determines if horizontal error bars are graphed with the xdata and ydata
                legend (bool, optional): determines if legend is graphed
                guesses (array_like, optional): starting guesses for fit function
                    see p0 parameter in scipy.optimize.curve_fit
                return_raw (bool, optional): if true will return Raw_parameters instead of Parameters

        Returns:
                Parameters (list): an array of strings containing the name of each parameter in the fit function along with its value and uncertanty
                Raw_Parameters (list): a list containing a list of values for the fit parameters and list of values for the uncertanty in the fit parameters
    '''

    ## Determine Fit parameters 
    if yerror is not None and fit is not None: #Calulates Fit Parameters when an error is provided
        
        if guesses is not None: #Calculates fit parameters with guesses
            parameters, covariance = opt.curve_fit(fit, xdata, ydata,sigma = yerror,p0=guesses)

        else:#Calculates fit parameters without guesses
            parameters, covariance = opt.curve_fit(fit, xdata, ydata,sigma = yerror)

        perr = np.sqrt(np.diag(covariance)) #calculates error in values based on the covariance matrix
        
    elif(fit is not None): #Calulates Fit Parameters when no error is provided
        
        if guesses is not None: #Calculates fit parameters with guesses
            parameters, covariance = opt.curve_fit(fit, xdata, ydata,p0=guesses)
            
        else:#Calculates fit parameters without guesses
            parameters, covariance = opt.curve_fit(fit, xdata, ydata)
            
        perr = np.sqrt(np.diag(covariance)) 
        

    if y_bar and not x_bar: #plots error bars
        if yerror is None:
            print("Error plotting error bars: y_bar= True but no error values were specified") 
        try:
            plt.errorbar(xdata,ydata,yerr=yerror,capsize = 5,marker = 'o',linestyle = 'None', label = dataname)
        except Exception as ex:
            # prints exception if there is an error while plotting error bars
            print(f"Error plotting error bars: {ex}")
    if x_bar and not y_bar: #plots error bars
        if xerror is None:
            print("Error plotting error bars: x_bar= True but no error values were specified") 
        try:
            plt.errorbar(xdata,ydata,xerr=xerror,capsize = 5,marker = 'o',linestyle = 'None', label = dataname)
        except Exception as ex:
            # prints exception if there is an error while plotting error bars
            print(f"Error plotting error bars: {ex}")
    if x_bar and y_bar:
        if xerror is None or yerror is None:
            print(f"Error plotting error bars: x_bar= {x_bar} and y_bar= {y_bar} but no error values were specified") 
        try:
            plt.errorbar(xdata,ydata,xerr=xerror,yerr=yerror,capsize = 5,marker = 'o',linestyle = 'None', label = dataname)
        except Exception as ex:
            # prints exception if there is an error while plotting error bars
            print(f"Error plotting error bars: {ex}")
        
    ## Plot data
    if fit is not None : #plots fit line and data
        try:
            plt.plot(xdata,fit(xdata,*parameters),label = fitname) 
            if not y_bar and not x_bar:
                plt.plot(xdata,ydata, 'o')

        except:
            print("an error occured, is your fit function correct?")

        #isolate parameter names
        param_info = inspect.getfullargspec(fit) 
        param_names = param_info[0][1:]

    else:
        plt.plot(xdata,ydata,'o',label = dataname,)
        
    #Labels Legend    
    if legend :
        plt.legend(loc = 'upper left')
    if xname is not None:
        plt.xlabel(xname, fontsize = 16)
    if yname is not None:
        plt.ylabel(yname, fontsize = 16)
    if title is not None:
        plt.title(title, fontsize = 18)

    plt.show()
    
    
    if not return_raw: #
        if(fit is not None):
            return [f"{param} = {parameters[param_names.index(param)]} ± {perr[param_names.index(param)]}" for param in param_names] 
    else:
        if(fit is not None):
            return [parameters,perr]
    #End Logic

