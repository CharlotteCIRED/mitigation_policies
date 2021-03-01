# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 14:05:38 2021

@author: Charlotte Liotta
"""

import numpy as np
import pandas as pd

def compute_emissions(density, grid, trans, scenar_emissions = None, initial_year = None, index = None):
    EMISSIONS_CAR = 100
    EMISSIONS_PUBLIC_TRANSPORT = 6
    
    if isinstance(scenar_emissions, pd.Series) == False:
        variation_emissions = 1
    else:
        variation_emissions = scenar_emissions[initial_year + index] / scenar_emissions[initial_year]
        
    print(variation_emissions)
    return 2*365 * ((np.nansum(density[trans.mode == 0] * grid.distance_cbd[trans.mode == 0]) * (EMISSIONS_CAR * variation_emissions)) + (np.nansum(density[trans.mode == 1] * grid.distance_cbd[trans.mode == 1]) * (EMISSIONS_PUBLIC_TRANSPORT * variation_emissions)))

def compute_housing_supply(housing_supply_t1_without_inertia, housing_supply_t0):
        TIME_LAG = 3
        DEPRECIATION_TIME = 100
        diff_housing = ((housing_supply_t1_without_inertia - housing_supply_t0) / TIME_LAG) - (housing_supply_t0 / DEPRECIATION_TIME)
        for i in range(0, len(housing_supply_t1_without_inertia)):
            if housing_supply_t1_without_inertia[i] <= housing_supply_t0[i]:
                diff_housing[i] = - (housing_supply_t0[i] / DEPRECIATION_TIME)

        housing_supply_t1 = housing_supply_t0 + diff_housing
        return housing_supply_t1
    
def import_city_scenarios(city, country):
    """ Import World Urbanization Prospects scenarios.
    
    Population growth rate at the city scale.
    """
    
    city = city.replace('_', ' ')
    city = city.replace('Ahmedabad', 'Ahmadabad')
    city = city.replace('Belem', 'Belém')
    city = city.replace('Bogota', 'Bogot')
    city = city.replace('Brasilia', 'Bras')
    city = city.replace('Brussels', 'Brussel')
    city = city.replace('Wroclaw', 'Wroc')
    city = city.replace('Valparaiso', 'Valpar')
    city = city.replace('Ulan Bator', 'Ulaanbaatar')
    city = city.replace('St Petersburg', 'Petersburg')
    city = city.replace('Sfax', 'Safaqis')
    city = city.replace('Seville', 'Sevilla')
    city = city.replace('Sao Paulo', 'Paulo')
    city = city.replace('Poznan', 'Pozna')
    city = city.replace('Porto Alegre', 'Alegre')
    city = city.replace('Nuremberg', 'Nurenberg')
    city = city.replace('Medellin', 'Medell')
    city = city.replace('Washington DC', 'Washington')
    city = city.replace('San Fransisco', 'San Francisco')
    city = city.replace('Rostov on Don', 'Rostov')
    city = city.replace('Nizhny Novgorod', 'Novgorod')
    city = city.replace('Mar del Plata', 'Mar Del Plata')
    city = city.replace('Malmo', 'Malm')
    city = city.replace('Lodz', 'Łódź')
    city = city.replace('Leeds', 'West Yorkshire')
    city = city.replace('Jinan', "Ji'nan")
    city = city.replace('Isfahan', 'Esfahan')
    city = city.replace('Hanover', 'Hannover')
    city = city.replace('Gothenburg', 'teborg')
    city = city.replace('Goiania', 'nia')
    city = city.replace('Ghent', 'Gent')
    city = city.replace('Geneva', 'Genève')
    city = city.replace('Fez', 'Fès')
    city = city.replace('Cluj Napoca', 'Cluj-Napoca')
    city = city.replace('Cordoba', 'rdoba')
    city = city.replace('Concepcion', 'Concepc')
    country = country.replace('_', ' ')
    country = country.replace('UK', 'United Kingdom')
    country = country.replace('Russia', 'Russian Federation')
    country = country.replace('USA', 'United States of America')
    country = country.replace('Czech Republic', 'Czechia')
    country = country.replace('Ivory Coast', 'Ivoire')
    
    scenario_growth_rate = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/WUP2018-F14-Growth_Rate_Cities.xls', 
                                         skiprows = 15, 
                                         header = 1)
    
    scenario_growth_rate = scenario_growth_rate.rename(
        columns={
            "Urban Agglomeration" : 'city', 
            "Country or area" : 'country'})
    
    growth_rate = {
        "2015-2020" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2015-2020'].squeeze(),
        "2020-2025" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2020-2025'].squeeze(),
        "2025-2030" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2025-2030'].squeeze(),
        "2030-2035" : scenario_growth_rate.loc[(scenario_growth_rate.city.str.find(city) != -1) & (scenario_growth_rate.country.str.find(country) != -1), '2030-2035'].squeeze()}
    

    return growth_rate

def import_country_scenarios(country):
    """ Import World Urbanization Prospects scenarios.
    
    Urban opulation growth rate at the country scale.
    To be used when data at the city scale are not available.
    """
    
    country = country.replace('_', ' ')
    
    
    scenario_growth_rate = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/WUP2018-F06-Urban_Growth_Rate.xls', 
                                         skiprows = 14, 
                                         header = 1)
    
    scenario_growth_rate = scenario_growth_rate.rename(
        columns = {
            'Unnamed: 1' : 'country', 
            'Unnamed: 17': '2015-2020', 
            'Unnamed: 18': '2020-2025', 
            'Unnamed: 19': '2025-2030', 
            'Unnamed: 20': '2030-2035'})
    
    growth_rate = {
        "2015-2020" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2015-2020'].squeeze(),
        "2020-2025" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2020-2025'].squeeze(),
        "2025-2030" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2025-2030'].squeeze(),
        "2030-2035" : scenario_growth_rate.loc[(scenario_growth_rate.country.str.find(country) != -1), '2030-2035'].squeeze()}
       
    return growth_rate

def compute_r2(dataCity, simul_density, simul_rent, simul_size, selected_cells):
    """ Explained variance / Total variance """
    
    sst_rent = ((dataCity.rent[selected_cells] - 
                 np.nanmean(dataCity.rent[selected_cells])) ** 2).sum()
    
    sst_density = (((dataCity.density[selected_cells] - 
                 np.nanmean(dataCity.density[selected_cells])) ** 2).sum())
    
    sst_size = ((dataCity.size[selected_cells] - 
                 np.nanmean(dataCity.size[selected_cells])) ** 2).sum()
    
    sse_rent = ((dataCity.rent[selected_cells] - 
                 simul_rent[selected_cells]) ** 2).sum()
    
    sse_density = (((dataCity.density[selected_cells] - 
                 simul_density[selected_cells]) ** 2).sum())
    
    sse_size = ((dataCity.size[selected_cells] - 
                 simul_size[selected_cells]) ** 2).sum()
    
    r2_rent = 1 - (sse_rent / sst_rent)
    r2_density = 1 - (sse_density / sst_density)
    r2_size = 1 - (sse_size / sst_size)
    
    return r2_rent, r2_density, r2_size
