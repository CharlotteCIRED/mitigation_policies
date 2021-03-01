# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:21:27 2021

@author: Charlotte Liotta
"""

# %% loop on cities
d_population = {}
d_income = {}
d_car_speed = {}
d_transit_speed = {}

d_growth_rate_2015_2020 = {}
d_growth_rate_2020_2025 = {}
d_growth_rate_2025_2030 = {}
d_growth_rate_2030_2035 = {}

d_coeff_land = {}
d_grid = {}


for city in np.unique(list_city.City):#[166:192]:[[17,144]][[17,63,107,144]]
    #NB: try, c'est du tres mauvais codage de ma part: le code s'execute meme s'il y a une erreur. A enlever...
    print("\n*** " + city + " ***\n")

    print("\n** Import data **\n")

    #Import data
    (country, density, rents_and_size, land_use, income, conversion_rate, driving, transit, grille, centre, distance_cbd) = import_data(list_city, paths_data, city)

    #explication: driving.Distance c'est en metres, on part sur une conso de 7.18 l/100km et 0.860 €/l d'essence
    prix_driving=driving.Duration*income/ (3600 * 24)/365+driving.Distance*0.860*7.18/100000
    prix_transit=transit.Duration*income/ (3600 * 24)/365+transit.Distance*0.860*7.18/100000
    #si que cout du temps
    # prix_driving=driving.Duration*income/ (3600 * 8)
    # prix_transit=transit.Duration*income/ (3600 * 8)

    tous_prix=np.vstack((prix_driving,prix_transit))#les prix concaténés
    prix_transport=np.amin(tous_prix, axis=0)
    prix_transport[np.isnan(prix_transit)]=prix_driving[np.isnan(prix_transit)]
    prix_transport=prix_transport*2*365/income # on l'exprime par rapport au revenu
    prix_transport=pd.Series(prix_transport)
    mode_choice=np.argmin(tous_prix, axis=0)
    mode_choice[np.isnan(prix_transit)]=0

    #Put them into a "city" structure
    dataCity = CityClass(
        density = density.loc[:,density.columns.str.startswith("density")].squeeze(),
        rent = (rents_and_size.medRent / conversion_rate) * 12,
        size = rents_and_size.medSize,#
        urb = (land_use.OpenedToUrb / land_use.TotalArea) * COEFF_URB,
        income = income,
        duration = prix_transport,
        duration_driving=driving.Duration,
        distance_driving=driving.Distance,
        duration_transit=transit.Duration,
        distance_transit=transit.Distance,
        mode_choice=mode_choice
        )

    dataCity.total_population=np.nansum(dataCity.density)
    
    d_population[city] = dataCity.total_population
    d_income[city] = dataCity.income
    d_car_speed[city] = np.nansum((driving.Distance / driving.Duration) * dataCity.density) / np.nansum(dataCity.density)
    d_transit_speed[city] = np.nansum((transit.Distance / transit.Duration) * dataCity.density) / np.nansum(dataCity.density)
    d_coeff_land[city] = dataCity.urb
    d_grid[city] = grille
    
    d_growth_rate_2015_2020[city] = import_city_scenarios(city, country, paths_data)['2015-2020']
    d_growth_rate_2020_2025[city] = import_city_scenarios(city, country, paths_data)['2020-2025']
    d_growth_rate_2025_2030[city] = import_city_scenarios(city, country, paths_data)['2025-2030']
    d_growth_rate_2030_2035[city] = import_city_scenarios(city, country, paths_data)['2030-2035']

np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_population.npy', d_population) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_income.npy', d_income) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_car_speed.npy', d_car_speed) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_transit_speed.npy', d_transit_speed) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_coeff_land.npy', d_coeff_land) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_grid.npy', d_grid) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_growth_rate_2015_2020.npy', d_growth_rate_2015_2020) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_growth_rate_2020_2025.npy', d_growth_rate_2020_2025) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_growth_rate_2025_2030.npy', d_growth_rate_2025_2030) 
np.save('C:/Users/Charlotte Liotta/Desktop/these/Sorties/database/d_growth_rate_2030_2035.npy', d_growth_rate_2030_2035) 

plt.scatter(d_grid["Rio_de_Janeiro"].XCOORD,d_grid["Rio_de_Janeiro"].YCOORD,d_coeff_land["Rio_de_Janeiro"])