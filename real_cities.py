# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:45:51 2021

@author: Coupain
"""


def import_paris():
    
    paris = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Parisfilename.pickle", "rb", -1))
    grid_paris = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Parisgrid.pickle", "rb", -1))
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_paris.npy',allow_pickle='TRUE').item()
    param_paris = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : paris.urb,
        "grid" : grid_paris,
        "pop_growth_rate" : [import_city_scenarios('Paris', 'France', paths_data)['2015-2020'], import_city_scenarios('Paris', 'France', paths_data)['2020-2025'], import_city_scenarios('Paris', 'France', paths_data)['2025-2030'], import_city_scenarios('Paris', 'France', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : paris.income,
        "init_population" : paris.total_population,
        "duration_driving" : paris.duration_driving,
        "duration_transit" : paris.duration_transit,
        "distance_driving" : paris.distance_driving,
        "distance_transit" : paris.distance_transit,
        "data_density": paris.density,
        "data_rent" : paris.rent,
        "data_size" : paris.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_paris

def import_atlanta():
    atlanta = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Atlantafilename.pickle", "rb", -1))
    grid_atlanta = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Atlantagrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_atlanta.npy',allow_pickle='TRUE').item()
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_atlanta = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : atlanta.urb,
        "grid" : grid_atlanta,
        "pop_growth_rate" : [import_city_scenarios('Atlanta', 'United States of America', paths_data)['2015-2020'], import_city_scenarios('Atlanta', 'United States of America', paths_data)['2020-2025'], import_city_scenarios('Atlanta', 'United States of America', paths_data)['2025-2030'], import_city_scenarios('Atlanta', 'United States of America', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : atlanta.income,
        "init_population" : atlanta.total_population,
        "duration_driving" : atlanta.duration_driving,
        "duration_transit" : atlanta.duration_transit,
        "distance_driving" : atlanta.distance_driving,
        "distance_transit" : atlanta.distance_transit,
        "data_density": atlanta.density,
        "data_rent" : atlanta.rent,
        "data_size" : atlanta.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_atlanta

def import_sao_paolo():
    sao_paolo = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Sao_Paulofilename.pickle", "rb", -1))
    grid_sao_paolo = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Sao_Paulogrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_sao_paolo.npy',allow_pickle='TRUE').item()
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_sao_paolo = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : sao_paolo.urb,
        "grid" : grid_sao_paolo,
        "pop_growth_rate" : [import_city_scenarios('Paulo', 'Brazil', paths_data)['2015-2020'], import_city_scenarios('Paulo', 'Brazil', paths_data)['2020-2025'], import_city_scenarios('Paulo', 'Brazil', paths_data)['2025-2030'], import_city_scenarios('Paulo', 'Brazil', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : sao_paolo.income,
        "init_population" : sao_paolo.total_population,
        "duration_driving" : sao_paolo.duration_driving,
        "duration_transit" : sao_paolo.duration_transit,
        "distance_driving" : sao_paolo.distance_driving,
        "distance_transit" : sao_paolo.distance_transit,
        "data_density": sao_paolo.density,
        "data_rent" : sao_paolo.rent,
        "data_size" : sao_paolo.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_sao_paolo

def import_shanghai():
    shanghai = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Shanghaifilename.pickle", "rb", -1))
    grid_shanghai = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Shanghaigrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_shanghai.npy',allow_pickle='TRUE').item()
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_shanghai = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : shanghai.urb,
        "grid" : grid_shanghai,
        "pop_growth_rate" : [import_city_scenarios('Shanghai', 'China', paths_data)['2015-2020'], import_city_scenarios('Shanghai', 'China', paths_data)['2020-2025'], import_city_scenarios('Shanghai', 'China', paths_data)['2025-2030'], import_city_scenarios('Shanghai', 'China', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : shanghai.income,
        "init_population" : shanghai.total_population,
        "duration_driving" : shanghai.duration_driving,
        "duration_transit" : shanghai.duration_transit,
        "distance_driving" : shanghai.distance_driving,
        "distance_transit" : shanghai.distance_transit,
        "data_density": shanghai.density,
        "data_rent" : shanghai.rent,
        "data_size" : shanghai.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    
    return param_shanghai

def import_curitiba():
    curitiba = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Curitibafilename.pickle", "rb", -1))
    grid_curitiba = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Curitibagrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_curitiba.npy',allow_pickle='TRUE').item()
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_curitiba = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : curitiba.urb,
        "grid" : grid_curitiba,
        "pop_growth_rate" : [import_city_scenarios('Curitiba', 'Brazil', paths_data)['2015-2020'], import_city_scenarios('Curitiba', 'Brazil', paths_data)['2020-2025'], import_city_scenarios('Curitiba', 'Brazil', paths_data)['2025-2030'], import_city_scenarios('Curitiba', 'Brazil', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : curitiba.income,
        "init_population" : curitiba.total_population,
        "duration_driving" : curitiba.duration_driving,
        "duration_transit" : curitiba.duration_transit,
        "distance_driving" : curitiba.distance_driving,
        "distance_transit" : curitiba.distance_transit,
        "data_density": curitiba.density,
        "data_rent" : curitiba.rent,
        "data_size" : curitiba.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_curitiba

def import_medellin():
    medellin = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Medellinfilename.pickle", "rb", -1))
    grid_medellin = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Medellingrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_medellin.npy',allow_pickle='TRUE').item()
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_medellin = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : medellin.urb,
        "grid" : grid_medellin,
        "pop_growth_rate" : [import_city_scenarios('Medellin', 'Colombia', paths_data)['2015-2020'], import_city_scenarios('Medellin', 'Colombia', paths_data)['2020-2025'], import_city_scenarios('Medellin', 'Colombia', paths_data)['2025-2030'], import_city_scenarios('Medellin', 'Colombia', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : medellin.income,
        "init_population" : medellin.total_population,
        "duration_driving" : medellin.duration_driving,
        "duration_transit" : medellin.duration_transit,
        "distance_driving" : medellin.distance_driving,
        "distance_transit" : medellin.distance_transit,
        "data_density": medellin.density,
        "data_rent" : medellin.rent,
        "data_size" : medellin.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_medellin

def import_kazan():
    kazan = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Kazanfilename.pickle", "rb", -1))
    grid_kazan = pickle.load(open("C:/Users/Charlotte Liotta/Desktop/these/Sorties/Kazangrid.pickle", "rb", -1))

    calibratedParam = np.load('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/calibratedParameters_kazan.npy',allow_pickle='TRUE').item()
    
    path_data='C:/Users/Charlotte Liotta/Desktop/eco_urbaine/2. Donnees/'
    paths_data={
        "path_data" : path_data,
        }
    param_kazan = {
        "beta" : calibratedParam['beta'],
        "A" : calibratedParam['kappa'],
        "b" : calibratedParam['b'],
        "Ro": calibratedParam['Ro'],
        "coeff_land" : kazan.urb,
        "grid" : grid_kazan,
        "pop_growth_rate" : [import_city_scenarios('Kazan', 'Russia', paths_data)['2015-2020'], import_city_scenarios('Kazan', 'Russia', paths_data)['2020-2025'], import_city_scenarios('Kazan', 'Russia', paths_data)['2025-2030'], import_city_scenarios('Kazan', 'Russia', paths_data)['2030-2035']],
        "income_growth_rate": 1,
        "income" : kazan.income,
        "init_population" : kazan.total_population,
        "duration_driving" : kazan.duration_driving,
        "duration_transit" : kazan.duration_transit,
        "distance_driving" : kazan.distance_driving,
        "distance_transit" : kazan.distance_transit,
        "data_density": kazan.density,
        "data_rent" : kazan.rent,
        "data_size" : kazan.size,
        "delta" : 0.05,
        "depreciation_time" : 100,
        "time_lag" : 3,
        "emissions_car" : 100, #gCO2/voy/km
        "emissions_public_transport" : 6}#gCO2/voy/km
    return param_kazan

def import_city_scenarios(city, country, paths_data):
    """ Import World Urbanization Prospects scenarios.
    
    Population growth rate at the city scale.
    """
    
    path_data=paths_data["path_data"]
    
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
    
    scenario_growth_rate = pd.read_excel(path_data + 
                                         'scenarios_wup/WUP2018-F14-Growth_Rate_Cities.xls', 
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


