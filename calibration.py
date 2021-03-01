# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:52:09 2021

@author: Charlotte Liotta
"""

import numpy as np
from scipy import optimize
import copy

def calibration(dataCity, trans, INTEREST_RATE, selected_cells, HOUSEHOLD_SIZE):
    
    bounds = ((0.1,0.99), #beta
              (0.001,None), #Ro
              (0.0,0.95), #b
              (0, None) #kappa
              )
    
    X0 = np.array([0.25, #beta
                   300, #Ro
                   0.64, #b
                   2] )  
    
    def minus_log_likelihood(X0):
    
        (simul_rent, 
         simul_dwelling_size, 
         simul_density) = model(X0, 
                                dataCity,
                                trans,
                                INTEREST_RATE, 
                                selected_cells, 
                                HOUSEHOLD_SIZE)
    
        (sum_ll,
         ll_R,
         ll_D,
         ll_Q,
         detail_ll_R,
         detail_ll_D,
         detail_ll_Q) = log_likelihood(simul_rent,
                                       simul_dwelling_size,
                                       simul_density,
                                       dataCity, 
                                       selected_cells)
    
        return -sum_ll
        
    result_calibration = optimize.minimize(minus_log_likelihood, X0, bounds=bounds) 
    
    return result_calibration

def model(X0, dataCity, trans, INTEREST_RATE, selected_cells, HOUSEHOLD_SIZE, housing_supply = None, OPTION = None):
    """ Compute rents, densities and dwelling sizes """    

    ## Starting point
    beta = X0[0]
    Ro = X0[1]
    b = X0[2]
    kappa = X0[3]
    
    a=1-b
    
    #we convert all panda dataframes to numpy so that everything is quicker
    trans_price = trans.transport_price.to_numpy()
    urb = dataCity.urb.to_numpy()
    income = dataCity.income  
    
    ## On simule les loyers, tailles des logements et densites
    #log_simul_rent = np.log(Ro) + (1 / beta) * (np.log(1 - (trans_price[selected_cells] / income)))
    #log_simul_size = np.log(HOUSEHOLD_SIZE) + np.log(beta * (income-(trans_price[selected_cells]))) - (log_simul_rent)
    #warning, the truye formulla is (1/a*np.log(kappa)
    #I changed it also in declare_structure
    #log_simul_density = ((1/a)*np.log(kappa)) + ((b / a) * (np.log(b / INTEREST_RATE))) + ((b / a) * (log_simul_rent)) - (log_simul_size - np.log(HOUSEHOLD_SIZE))+ np.log(urb[selected_cells])

    income_net_of_transport_costs = np.fmax(income - trans_price, np.zeros(len(trans_price)))             
    rent = (Ro * income_net_of_transport_costs**(1/beta) /income**(1/beta))
    np.seterr(divide = 'ignore', invalid = 'ignore')
    dwelling_size = beta * income_net_of_transport_costs / rent
    dwelling_size[rent == 0] = 0
    np.seterr(divide = 'warn', invalid = 'warn')
    #dwelling_size[np.isnan(dwelling_size)] = 0
    #dwelling_size[dwelling_size > 300] = 300
    if OPTION == 1:
        housing = copy.deepcopy(housing_supply)
    else:
        housing = urb * ((kappa**(1/a)) * (((b / INTEREST_RATE) * rent) ** (b/(a))))
    np.seterr(divide = 'ignore', invalid = 'ignore')
    density = copy.deepcopy(housing / dwelling_size)
    density[dwelling_size == 0] = 0
    np.seterr(divide = 'warn', invalid = 'warn')        
    density[np.isnan(density)] = 0    
    density[density == 0] = 1
    dwelling_size[dwelling_size == 0] = 1
    rent[rent == 0] = 1
    housing[np.isinf(housing)] = 10000000
    return rent, dwelling_size, density

def log_likelihood(simul_rent, simul_dwelling_size, simul_density, dataCity, selected_cells):
    """ Compute Log-Likelihood on rents, density and dwelling size based on model oputputs. """
    
    data_rent = dataCity.rent.to_numpy()
    data_density = dataCity.density.to_numpy()
    data_size = dataCity.size.to_numpy()

    x_R = (np.log(data_rent[selected_cells])) - (np.log(simul_rent[selected_cells]))
    x_Q = (np.log(data_size[selected_cells])) - (np.log(simul_dwelling_size[selected_cells]))
    x_D = (np.log(data_density[selected_cells])) - (np.log(simul_density[selected_cells]))
    
    sigma_r2 = (1/sum(selected_cells)) * np.nansum(x_R ** 2)
    sigma_q2 = (1/sum(selected_cells)) * np.nansum(x_Q ** 2)
    sigma_d2 = (1/sum(selected_cells)) * np.nansum(x_D ** 2)
        
    (ll_R, detail_ll_R) = ll_normal_distribution(x_R, sigma_r2)
    (ll_Q, detail_ll_Q) = ll_normal_distribution(x_Q, sigma_q2)
    (ll_D, detail_ll_D) = ll_normal_distribution(x_D, sigma_d2)
    
    return (ll_R + ll_Q + ll_D,
            ll_R,
            ll_D,
            ll_Q,
            detail_ll_R,
            detail_ll_D,
            detail_ll_Q)

def ll_normal_distribution(error, sigma2):
    """ normal distribution probability density function """
    
    log_pdf = -(error ** 2)/(2 * (sigma2))-1/2*np.log(sigma2)-1/2*np.log(2 * np.pi)
    
    return (np.nansum(log_pdf),log_pdf)


    