# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 10:27:37 2021

@author: Charlotte Liotta
"""

import numpy as np

def import_carbon_tax(duration):
    param_policies = {
        "carbon_tax_value" : 0.23, #euros per km
        "carbon_tax_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([1], duration - 5)]), #1 - 33
        "greenbelt_coeff" : 0,
        "greenbelt_threshold" : 2000,
        "greenbelt_start" : 100, #1 - 33
        "public_transport_speed_factor" : 1.2,
        "public_transport_speed_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)])
        }
    
    return param_policies

def import_greenbelt(duration):
    param_policies = {
        "carbon_tax_value" : 0.23, #euros per km
        "carbon_tax_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)]), #1 - 33
        "greenbelt_coeff" : 0,
        "greenbelt_threshold" : 2000,
        "greenbelt_start" : 5, #1 - 33
        "public_transport_speed_factor" : 1.2,
        "public_transport_speed_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)])
        }
    
    return param_policies

def import_public_transport_speed(duration):
    param_policies = {
        "carbon_tax_value" : 0.23, #euros per km
        "carbon_tax_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)]), #1 - 33
        "greenbelt_coeff" : 0,
        "greenbelt_threshold" : 2000,
        "greenbelt_start" : 100, #1 - 33
        "public_transport_speed_factor" : 1.2,
        "public_transport_speed_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([1], duration - 5)])
        }
    
    return param_policies

def import_no_policy(duration):
    param_policies = {
        "carbon_tax_value" : 0.23, #euros per km
        "carbon_tax_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)]), #1 - 33
        "greenbelt_coeff" : 0,
        "greenbelt_threshold" : 2000,
        "greenbelt_start" : 100, #1 - 33
        "public_transport_speed_factor" : 1.2,
        "public_transport_speed_implementation" : np.concatenate([np.repeat([0], 5), np.repeat([0], duration - 5)])
        }
    
    return param_policies