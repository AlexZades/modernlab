import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import inspect as inspect

#Calcuates error in functions variationally
def quick_plot(xdata, ydata, xname = None, yname = None, title = None, linename = None, yerror = None, xerror = None, fit = None, y_bar = False, x_bar = False, legend = False, guesses = None):

        
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
        
        
    if y_bar: #plots error bars
        if yerror is None:
            print("Error plotting error bars: y_bar= True but no error values were specified") 
        try:
            plt.errorbar(xdata,ydata,yerr=yerror,capsize = 5,marker = 'o',linestyle = 'None', label = 'data')
        except Exception as ex:
            # prints exception if there is an error while plotting error bars
            print(f"Error plotting error bars: {ex}")
    if x_bar: #plots error bars
        if xerror is None:
            print("Error plotting error bars: x_bar= True but no error values were specified") 
        try:
            plt.errorbar(xdata,ydata,xerr=xerror,capsize = 5,marker = 'o',linestyle = 'None', label = 'data')
        except Exception as ex:
            # prints exception if there is an error while plotting error bars
            print(f"Error plotting error bars: {ex}")
        
    ## Plot data
    if fit is not None : #plots fit line and data
        try:
            plt.plot(xdata,fit(xdata,*parameters),label = linename) 
            if not y_bar:
                plt.plot(xdata,ydata, 'o')

        except:
            print("an error occured, is your fit function correct?")

        #isolate parameter names
        param_info = inspect.getfullargspec(fit) 
        param_names = param_info[0][1:]

    else:
        plt.plot(xdata,ydata,'o',label = linename,)
        
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
    
    
    
    if (yerror is not None and fit is not None) or (xerror is not None and fit is not None) : 
        #returns parameter
        return [f"{param} = {parameters[param_names.index(param)]} +/-{perr[param_names.index(param)]}" for param in param_names]
    elif(fit is not None):
        return [f"{param} = {parameters[param_names.index(param)]} +/-{perr[param_names.index(param)]}" for param in param_names]

    else:
        return 0


def variational_error(func,params,error,param_id):
    
    values = []
     #converts error and param_id to lists if only a single value is provided
        
    if type(error) != list and type(error) != np.array:
        error = [error]
        param_id = [param_id]
        
    # adds error to each value spesified in param_id
    
    for i in param_id:
        values.append(params[i]) 
        params[i] = values[param_id.index(i)] + error[param_id.index(i)]
        
    # calculates high value of function
    up_calc = np.abs(func(*params))
    
    # subtracts error from each value spesified in param_id
    for i in param_id:
        params[i] = values[param_id.index(i)] - error[param_id.index(i)]
    
    # calculates low value of function
    down_calc = np.abs(func(*params))
    
    #returns the average of the difference between the high and low values
    return np.abs(up_calc - down_calc)/2

