# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:40:31 2021

@author: Charlotte Liotta
"""

import pandas as pd 
import numpy as np
import timeit #pour mesurer les temps
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #pour les grapohes en 3d
import scipy as sc
from scipy import optimize
import math
import os
import pickle
import seaborn as sns

#from structures import *
from functions import *
#from real_cities import *
from import_policies import *
from calibration import *
from commons import declare_structures


#city = 'Paris'
duration = 21
arr = os.listdir("C:/Users/Coupain/Desktop/these/Sorties/data")
policy = import_no_policy(duration)
policy_name = "no_policy"
#path_outputs = "C:/Users/Coupain/Desktop/these/Sorties/" + city + policy_name
#os.mkdir(path_outputs)
HOUSEHOLD_SIZE = 1
INTEREST_RATE = 0.05
beta = {}
b = {}
Ro = {}
kappa = {}

r2rent = {}
r2density = {}
r2size = {}

beta = np.load("C:/Users/Coupain/Desktop/these/Sorties/validation/beta.npy", allow_pickle = True).item()
Ro = np.load("C:/Users/Coupain/Desktop/these/Sorties/validation/Ro.npy", allow_pickle = True).item()
b = np.load("C:/Users/Coupain/Desktop/these/Sorties/validation/b.npy", allow_pickle = True).item()
kappa = np.load("C:/Users/Coupain/Desktop/these/Sorties/validation/kappa.npy", allow_pickle = True).item()

for index in range(0, 192):
    
    
    ### STEP 0 : PARAMETERS AND DATA

    city = arr[index * 2].replace('filename.pickle', '')
    print(city)
    country = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").country[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    region = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").region[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    dataCity = pickle.load(open("C:/Users/Coupain/Desktop/these/Sorties/data/" + city + "filename.pickle", "rb", -1))
    grid = pickle.load(open("C:/Users/Coupain/Desktop/these/Sorties/data/" + city + "grid.pickle", "rb", -1))
    trans = TransportSimulation()
    trans.create_trans(dataCity, policy, 0)

    ##### STEP 1: CALIBRATION

    selected_cells = np.array(dataCity.duration.notnull() & (dataCity.duration!=0)
                              & dataCity.urb.notnull()  & (dataCity.urb!=0)
                              & dataCity.rent.notnull()  & (dataCity.rent!=0)
                              & dataCity.size.notnull()  & (dataCity.size!=0)
                              & dataCity.density.notnull() & (dataCity.density!=0)
                              )

    #result_calibration = calibration(dataCity,
    #                                 trans,
    #                                 INTEREST_RATE,
    #                                 selected_cells,
    #                                 HOUSEHOLD_SIZE)

    #calibratedParameters = {
    #    "beta" : result_calibration.x[0],
    #    "Ro" : result_calibration.x[1],
    #    "b" : result_calibration.x[2],
    #    "kappa" : result_calibration.x[3],
    #    "HouseholdSize" : HOUSEHOLD_SIZE
    #    }

    #beta[city] = calibratedParameters["beta"]
    #b[city] = calibratedParameters["b"]
    #Ro[city] = calibratedParameters["Ro"]
    #kappa[city] = calibratedParameters["kappa"]

    #rent, dwelling_size, density = model(result_calibration.x,
    #                                     dataCity,
    #                                     trans,
    #                                     INTEREST_RATE, 
    #                                     selected_cells, 
    #                                     HOUSEHOLD_SIZE)

    init = np.array([beta[city], Ro[city], b[city], kappa[city]])
    rent, dwelling_size, density = model(init,
                                         dataCity,
                                         trans,
                                         INTEREST_RATE, 
                                         selected_cells, 
                                         HOUSEHOLD_SIZE)
    
    #(r2_rent, r2_density, r2_size) = compute_r2(dataCity, density, rent, dwelling_size, selected_cells)
    #r2rent[city] = r2_rent
    #r2density[city] = r2_density
    #r2size[city] = r2_size
    
    #plt.scatter(grid.distance_cbd, rent, label = "Simul", s=2, c="darkred")
    #plt.scatter(grid.distance_cbd, dataCity.rent, label = "Data", s=2, c="grey")
    #plt.legend(markerscale=6,
    #           scatterpoints=1, fontsize=10)
    #plt.title("Rent calibration")
    #plt.savefig("C:/Users/Coupain/Desktop/these/Sorties/validation/" + city + "_rent.png")
    #plt.close()

    #plt.scatter(grid.distance_cbd, density, label = "Simul", s=2, c="darkred")
    #plt.scatter(grid.distance_cbd, dataCity.density, label = "Data", s=2, c="grey")
    #plt.legend(markerscale=6,
    #           scatterpoints=1, fontsize=10)
    #plt.title("Density calibration")
    #plt.savefig("C:/Users/Coupain/Desktop/these/Sorties/validation/" + city + "_density.png")
    #plt.close()

    #plt.scatter(grid.distance_cbd, dwelling_size, label = "Simul", s=2, c="darkred")
    #plt.scatter(grid.distance_cbd, dataCity.size, label = "Data", s=2, c="grey")
    #plt.legend(markerscale=6,
    #           scatterpoints=1, fontsize=10)
    #plt.title("Dwelling size calibration")
    #plt.savefig("C:/Users/Coupain/Desktop/these/Sorties/validation/" + city + "_size.png")
    #plt.close()

#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/beta.npy", beta) 
#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/b.npy" , b) 
#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/Ro.npy", Ro) 
#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/kappa.npy", kappa) 

#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/r2rent.npy" , r2rent) 
#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/r2density.npy", r2density) 
#np.save("C:/Users/Coupain/Desktop/these/Sorties/validation/r2size.npy", r2size) 

#sns.distplot(list(beta.values()), kde = False, bins = 10)
#plt.ylabel("Number of cities")
#sns.distplot(list(b.values()), kde = False, bins = 10)
#plt.ylabel("Number of cities")
#sns.distplot(list(Ro.values()), kde = False, bins = 10)
#plt.ylabel("Number of cities")
#sns.distplot(np.array(list(kappa.values()))[np.array(list(kappa.values())) < 0.2], kde = False, bins = 10)
#plt.ylabel("Number of cities")

#for index in range(0, 192):
#    city = arr[index * 2].replace('filename.pickle', '')
#    plt.scatter(r2rent[city], r2density[city], c = 'steelblue')
#    plt.xlim(-1, 1)
#    plt.ylim(-1, 1)
#plt.hlines(0, -1, 1, colors = "darkgrey")
#plt.vlines(0, -1, 1, colors = "darkgrey")
#plt.ylabel("R2 density")
#plt.xlabel("R2 rent")

    #Residuals
    residuals = Residuals(
        density_residual = np.log(dataCity.density / density),
        rent_residual = np.log(dataCity.rent / rent),
        size_residual = np.log(dataCity.size / dwelling_size))
            
    residuals_for_simulation = Residuals(
        density_residual = np.where((dataCity.density.isnull()), 0, residuals.density_residual),
        rent_residual = np.where((dataCity.rent.isnull()), 0, residuals.rent_residual),
        size_residual = np.where((dataCity.size.isnull()), 0, residuals.size_residual))
    
    residuals_for_simulation.density_residual = np.where(
        dataCity.density == 0, 
        np.log(1 / density), 
        residuals_for_simulation.density_residual)
    residuals_for_simulation.density_residual = np.where(
        density == 0, 
        np.log(dataCity.density), 
        residuals_for_simulation.density_residual)
    residuals_for_simulation.density_residual = np.where(
        (dataCity.density == 0) & (density == 0), 
        0, 
        residuals_for_simulation.density_residual)
    
    #if city == 'Medan':
    #    residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 9.5] = 0
    #    #Il faudrait mettre 8.4 pour être sûrs
    #    
    #if city == 'Abidjan':
    #    residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 8] = 0
   # 
    #if city == 'Karachi':
    #    residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 7.5] = 0
    # 
    #if city == 'Jakarta':
    #    residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 7.5] = 0
    # 
    #if city == 'Bandung':
    #    residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 7.5] = 0
    # 
    #residuals_for_simulation.density_residual[residuals_for_simulation.density_residual > 7.5] = 0
    
    ##### STEP 2: SIMULATION

    #carbon_tax = import_carbon_tax(duration)
    #greenbelt = import_greenbelt(duration)
    #public_transport_speed = import_public_transport_speed(duration)
    #no_policy = import_no_policy(duration)
    
    policy = import_no_policy(duration)

    # %% PLOT

    #def map2D(x):
    #    plt.scatter(grid.coord_X, grid.coord_Y, s = None, c = x, marker='.', cmap=plt.cm.RdYlGn)
    #    cbar = plt.colorbar()  
    #    plt.show()

    #def map3D(x):
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')    
    #    ax.scatter(grid.coord_X, grid.coord_Y, x, c = x, alpha = 0.2, marker='.')    
    #    ax.set_xlabel('coord_X')
    #    ax.set_ylabel('coord_Y')
    #    ax.set_zlabel('Value')    
    #    plt.show()
        
    
    # %% EQUILIBRIUM

    def compute_residual(R_0):
        X0 = np.array([beta[city], R_0, b[city], kappa[city]])
        rent, dwelling_size, density = model(X0,
                                             dataCity,
                                             trans,
                                             INTEREST_RATE, 
                                             selected_cells, 
                                             HOUSEHOLD_SIZE)
        rent = rent * np.exp(residuals_for_simulation.rent_residual)
        density = density * np.exp(residuals_for_simulation.density_residual)
        dwelling_size = dwelling_size * np.exp(residuals_for_simulation.size_residual)
        delta_population = dataCity.total_population - np.nansum(density)
        return delta_population

    R_0 = sc.optimize.fsolve(compute_residual, Ro[city])
    X0 = np.array([beta[city], R_0, b[city], kappa[city]])
    rent, dwelling_size, density = model(X0,
                                         dataCity,
                                         trans,
                                         INTEREST_RATE, 
                                         selected_cells, 
                                         HOUSEHOLD_SIZE)

    rent = rent * np.exp(residuals_for_simulation.rent_residual)
    density = density * np.exp(residuals_for_simulation.density_residual)
    dwelling_size = dwelling_size * np.exp(residuals_for_simulation.size_residual)
    housing = dwelling_size * density

    #plt.scatter(grid.distance_cbd, dataCity.rent, label = "Data")
    #plt.scatter(grid.distance_cbd, rent, label = "Simul")
    #plt.legend()
    #plt.title("Rent calibration")
    #plt.close()

    #plt.scatter(grid.distance_cbd, dataCity.density, label = "Data")
    #plt.scatter(grid.distance_cbd, density, label = "Simul")
    #plt.legend()
    #plt.title("Density calibration")
    #plt.close()

    #plt.scatter(grid.distance_cbd, dataCity.size, label = "Data")
    #plt.scatter(grid.distance_cbd, dwelling_size, label = "Simul")
    #plt.legend()
    #plt.title("Dwelling size calibration")
    #plt.close()

    # %% DYNAMICS

    initial_year = 2015
    final_year = initial_year + duration - 1
    housing_supply_t0 = housing
    population = np.nansum(density)
    index = 0

    pop_growth_rate = import_city_scenarios(city, country)
    if isinstance(pop_growth_rate["2015-2020"], pd.Series) == True:
        pop_growth_rate = import_country_scenarios(country)
    imaclim = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx', sheet_name = 'Extraction_Baseline')
    imaclim = imaclim[imaclim.Region == region]
    scenar_income = imaclim[imaclim.Variable == "Index_income"].squeeze()
    scenar_driving_price = imaclim[imaclim.Variable == "Index_prix_auto"].squeeze()
    scenar_emissions = imaclim[imaclim.Variable == "Index_emissions_auto"].squeeze()

    save_emissions = np.zeros(duration)
    save_emissions_per_capita = np.zeros(duration)
    save_population = np.zeros(duration)
    save_R0 = np.zeros(duration)
    save_utility = np.zeros(duration)
    save_z = np.nan * np.empty((duration, len(grid.distance_cbd)))
    save_rent = np.nan * np.empty((duration, len(grid.distance_cbd)))
    save_dwelling_size = np.nan * np.empty((duration, len(grid.distance_cbd)))
    save_housing = np.nan * np.empty((duration, len(grid.distance_cbd)))
    save_density = np.nan * np.empty((duration, len(grid.distance_cbd)))
    save_income = np.zeros(duration)

    #First iteration
    
    save_emissions[index] = compute_emissions(density, grid, trans)
    save_emissions_per_capita[index] = save_emissions[index] / population
    save_population[index] = population
    save_R0[index] = R_0
    save_rent[index, :] = rent
    save_dwelling_size[index, :] = dwelling_size
    save_housing[index, :] = housing
    save_density[index, :] = density
    save_utility[index] = (dataCity.income  * ((1 - beta[city]) ** (1 - beta[city])) * (beta[city] ** beta[city])) / (save_R0[index] ** beta[city])
    
    save_income[index] = dataCity.income

    while initial_year + index < final_year:
    
        #Iterate and adjust
        index = index + 1
    
        if (initial_year + index >= 2015) & (initial_year + index < 2020):
            population = population * (1 + (pop_growth_rate["2015-2020"] / 100))
        elif (initial_year + index >= 2020) & (initial_year + index < 2025):
            population = population * (1 + (pop_growth_rate["2020-2025"] / 100))
        elif (initial_year + index >= 2025) & (initial_year + index < 2030):
            population = population * (1 + (pop_growth_rate["2025-2030"] / 100))
        elif (initial_year + index >= 2030):
            population = population * (1 + (pop_growth_rate["2030-2035"] / 100))
    
        dataCity.income = dataCity.income * (scenar_income[initial_year + index] / scenar_income[initial_year + index - 1])
    
        trans = TransportSimulation()
        trans.create_trans(dataCity, policy, index, scenar_driving_price, initial_year)
    
        if policy["greenbelt_start"] <= index:
                dataCity.urb[save_density[index - 1, :] <= policy["greenbelt_threshold"]] = policy["greenbelt_coeff"]
    
        #step 1 - without inertia
        def compute_residual(R_0):
            X0 = np.array([beta[city], 
                           R_0,
                           b[city], 
                           kappa[city]])
        
            rent, dwelling_size, density = model(X0,
                                                 dataCity,
                                                 trans,
                                                 INTEREST_RATE, 
                                                 selected_cells, 
                                                 HOUSEHOLD_SIZE)
        
            rent = rent * np.exp(residuals_for_simulation.rent_residual)
            density = density * np.exp(residuals_for_simulation.density_residual)
            dwelling_size = dwelling_size * np.exp(residuals_for_simulation.size_residual)

            delta_population = population - np.nansum(density)
            return delta_population

        R0_t1_without_inertia = sc.optimize.fsolve(compute_residual, Ro[city])
        X0 = np.array([beta[city], R0_t1_without_inertia, b[city], kappa[city]])
        (rent_t1_without_inertia, 
         dwelling_size_t1_without_inertia, 
         density_t1_without_inertia) = model(X0,
                                             dataCity,
                                             trans,
                                             INTEREST_RATE, 
                                             selected_cells, 
                                             HOUSEHOLD_SIZE)
                                         
        rent_t1_without_inertia = rent_t1_without_inertia * np.exp(residuals_for_simulation.rent_residual)
        density_t1_without_inertia = density_t1_without_inertia * np.exp(residuals_for_simulation.density_residual)
        dwelling_size_t1_without_inertia = dwelling_size_t1_without_inertia * np.exp(residuals_for_simulation.size_residual)
        housing_t1_without_inertia = density_t1_without_inertia * dwelling_size_t1_without_inertia
    
        #step 2 - with inertia   
        housing_supply_t1 = compute_housing_supply(housing_t1_without_inertia, housing_supply_t0)
        housing_supply_t1_without_amenities = housing_supply_t1 / (np.exp(residuals_for_simulation.size_residual)* np.exp(residuals_for_simulation.density_residual))
  
        def compute_residual(R_0):
            X0 = np.array([beta[city], R_0, b[city], kappa[city]])
            rent, dwelling_size, density = model(X0,
                                                 dataCity,
                                                 trans,
                                                 INTEREST_RATE, 
                                                 selected_cells, 
                                                 HOUSEHOLD_SIZE,
                                                 housing_supply_t1_without_amenities,
                                                 1)
            rent = rent * np.exp(residuals_for_simulation.rent_residual)
            density = density * np.exp(residuals_for_simulation.density_residual)
            dwelling_size = dwelling_size * np.exp(residuals_for_simulation.size_residual)      
            delta_population = population - np.nansum(density)
            return delta_population

        R0_t1 = sc.optimize.fsolve(compute_residual, R0_t1_without_inertia)
        X0 = np.array([beta[city], R0_t1, b[city], kappa[city]])
        rent_t1, dwelling_size_t1, density_t1 = model(X0,
                                                      dataCity,
                                                      trans,
                                                      INTEREST_RATE, 
                                                      selected_cells, 
                                                      HOUSEHOLD_SIZE,
                                                      housing_supply_t1_without_amenities,
                                                      1)
    
        rent_t1 = rent_t1 * np.exp(residuals_for_simulation.rent_residual)
        density_t1 = density_t1 * np.exp(residuals_for_simulation.density_residual)
        dwelling_size_t1 = dwelling_size_t1 * np.exp(residuals_for_simulation.size_residual)  
        housing_t1 = density_t1 * dwelling_size_t1
    
        #Export outputs
        save_emissions[index] = copy.deepcopy(compute_emissions(density_t1_without_inertia, grid, trans, scenar_emissions, initial_year, index))
        save_emissions_per_capita[index] = copy.deepcopy(save_emissions[index] / (population))
        save_population[index] = copy.deepcopy(population)
        save_R0[index] = copy.deepcopy(R0_t1_without_inertia)
        save_rent[index, :] = copy.deepcopy(rent_t1_without_inertia)
        save_dwelling_size[index, :] = copy.deepcopy(dwelling_size_t1_without_inertia)
        save_housing[index, :] = copy.deepcopy(housing_t1_without_inertia)
        save_density[index, :] = copy.deepcopy(density_t1_without_inertia)
        save_utility[index] = (dataCity.income  * ((1 - beta[city]) ** (1 - beta[city])) * (beta[city] ** beta[city])) / (save_R0[index] ** beta[city])
        save_z[index, :] = (dataCity.income  - rent_t1_without_inertia * dwelling_size_t1_without_inertia  - trans.transport_price)
        save_income[index] = dataCity.income     
        
        #Prepare iteration
        housing_supply_t0 = copy.deepcopy(housing_t1)
 
    ### EXPORT OUTPUTS
    plt.plot(save_emissions_per_capita)
    plt.xlabel("Year")
    plt.ylabel("Emissions per capita")
    plt.savefig("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_emissions_per_capita.png")
    plt.close()

    plt.plot(save_utility)
    plt.xlabel("Year")
    plt.ylabel("Utility")
    plt.savefig("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_utility.png")
    plt.close()

    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_emissions.npy", save_emissions)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_emissions_per_capita.npy", save_emissions_per_capita)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_population.npy", save_population)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_R0.npy", save_R0)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_rent.npy", save_rent)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_dwelling_size.npy", save_dwelling_size)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_housing.npy", save_housing)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_density.npy", save_density)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_utility.npy", save_utility)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_z.npy", save_z)
    np.save("C:/Users/Coupain/Desktop/these/Sorties/simulation/" + city + "_income.npy", save_income)
 
