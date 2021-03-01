# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:34:04 2021

@author: Coupain
"""

#### STEP 1: NO POLICY

#create the dataset
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    emissions_per_capita = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_no_policy/" + city + "_emissions_per_capita.npy")
    utility = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_no_policy/" + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    
no_policy = pd.DataFrame()
no_policy["city"] = emission_2015.keys()
no_policy["emissions_2015"] = emission_2015.values()
no_policy["emissions_2035"] = emission_2035.values()
no_policy["utility_2015"] = utility_2015.values()
no_policy["utility_2035"] = utility_2035.values()

#add continents
city_continent = pd.read_csv("C:/Users/Coupain/Desktop/these/Data/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
no_policy = no_policy.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

#plots
no_policy = no_policy.sort_values('emissions_2015')
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0,0,1,1])
ax.bar(no_policy.city[(no_policy.emissions_2015 < 400000) | (no_policy.emissions_2015 > 3000000)],no_policy.emissions_2015[(no_policy.emissions_2015 < 400000) | (no_policy.emissions_2015 > 3000000)])
plt.ylabel("gCO2/year")
plt.show()


color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
plt.figure(figsize=(20, 20))
plt.rcParams.update({'font.size': 25})
colors = list(no_policy['Continent'].unique())
for i in range(0 , len(colors)):
    data = no_policy.loc[no_policy['Continent'] == colors[i]]
    plt.scatter(data.utility_2015, data.emissions_2015, color=data.Continent.map(color_tab), label=colors[i], s = 200)
for i in range(192):
    plt.annotate(no_policy.city[i], (no_policy.utility_2015[i] + 100, no_policy.emissions_2015[i] + 25000))
plt.legend()
plt.xlabel("Utility")
plt.ylabel("Transport emissions per capita per year (gCO2)")
plt.show()

#### STEP 2: CARBON TAX

#create the dataset
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    emissions_per_capita = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_carbon_tax/" + city + "_emissions_per_capita.npy")
    utility = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_carbon_tax/" + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    
carbon_tax = pd.DataFrame()
carbon_tax["city"] = emission_2015.keys()
carbon_tax["emissions_2015"] = emission_2015.values()
carbon_tax["emissions_2035"] = emission_2035.values()
carbon_tax["utility_2015"] = utility_2015.values()
carbon_tax["utility_2035"] = utility_2035.values()

#add continents
city_continent = pd.read_csv("C:/Users/Coupain/Desktop/these/Data/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
carbon_tax = carbon_tax.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

#compare to baseline
baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
carbon_tax = carbon_tax.merge(baseline, left_on = "city", right_on = "city", how = 'left')
carbon_tax["emissions_reduction_2015"] = carbon_tax.emissions_2015 / carbon_tax.emissions_2015_BAU
carbon_tax["utility_reduction_2015"] = carbon_tax.utility_2015 / carbon_tax.utility_2015_BAU
carbon_tax["emissions_reduction_2035"] = carbon_tax.emissions_2035 / carbon_tax.emissions_2035_BAU
carbon_tax["utility_reduction_2035"] = carbon_tax.utility_2035 / carbon_tax.utility_2035_BAU

#plot

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
#plt.rcParams.update({'font.size': 25})
colors = list(carbon_tax['Continent'].unique())
plt.figure(figsize=(20, 20))
plt.xlim(-3, 0)
plt.ylim(-35, 0)
for i in range(0 , len(colors)):
    data = carbon_tax.loc[carbon_tax['Continent'] == colors[i]]
    plt.scatter((data.utility_reduction_2035 - 1) * 100, (data.emissions_reduction_2035 - 1) * 100, label=colors[i], s = 200, color=data.Continent.map(color_tab))
#for i in range(192):
#    plt.annotate(carbon_tax.city[i], (carbon_tax.utility_reduction_2035[i], carbon_tax.emissions_reduction_2035[i]))
plt.xlabel("Utility reduction compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.hlines(-10, -3, 0, colors = "black")
plt.vlines(-1, -35, 0, colors = "black")
plt.text(-0.5, -33, 'Appropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
plt.text(-2.9, -2, 'Inappropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
plt.legend()

sum((carbon_tax.emissions_reduction_2035 < 0.65) | (carbon_tax.utility_reduction_2035 < 0.97))
sum((carbon_tax.emissions_reduction_2035 > 1) | (carbon_tax.utility_reduction_2035 > 1))
#10 outliers !

recommended = carbon_tax.city[(carbon_tax.utility_reduction_2035 > 0.99) & (carbon_tax.emissions_reduction_2035 < 0.9)]
to_avoid = carbon_tax.city[(carbon_tax.utility_reduction_2035 < 0.99) & (carbon_tax.emissions_reduction_2035 > 0.9)]
big_impact = carbon_tax.city[(carbon_tax.utility_reduction_2035 < 0.99) & (carbon_tax.emissions_reduction_2035 < 0.9)]
small_impact = carbon_tax.city[(carbon_tax.utility_reduction_2035 > 0.99) & (carbon_tax.emissions_reduction_2035 > 0.9)]


### regression tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

df = copy.deepcopy(carbon_tax.iloc[:, [0,13,14]])
df["type"] = "A"
df.type[(df.utility_reduction_2035 < 0.99) & (df.emissions_reduction_2035 > 0.9)] = "D"
df.type[(df.utility_reduction_2035 > 0.99) & (df.emissions_reduction_2035 > 0.9)] = "B"
df.type[(df.utility_reduction_2035 < 0.99) & (df.emissions_reduction_2035 < 0.9)] = "C"
df = df.sort_values("city")

income = {}
population = {}
pop_growth = {}
income_growth = {}
car_speed = {}
transit_speed = {}

for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    dataCity = pickle.load(open("C:/Users/Coupain/Desktop/these/Sorties/data/" + city + "filename.pickle", "rb", -1))
    income[city] = dataCity.income
    population[city] = dataCity.total_population
    country = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").country[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    region = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").region[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    pop_growth_rate = import_city_scenarios(city, country)
    if isinstance(pop_growth_rate["2015-2020"], pd.Series) == True:
        pop_growth_rate = import_country_scenarios(country)
    pop_growth[city] = ((1+ (pop_growth_rate["2015-2020"]/100)) ** 5) * ((1+(pop_growth_rate["2020-2025"]/100)) ** 5) * ((1+(pop_growth_rate["2025-2030"]/100)) ** 5) * ((1+(pop_growth_rate["2030-2035"]/100)) ** 5)
    imaclim = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx', sheet_name = 'Extraction_Baseline')
    imaclim = imaclim[imaclim.Region == region]
    scenar_income = imaclim[imaclim.Variable == "Index_income"].squeeze()
    income_growth[city] = scenar_income[2035]/scenar_income[2015]
    car_speed[city] = np.nansum((dataCity.distance_driving / dataCity.duration_driving) * dataCity.density) / np.nansum(dataCity.density)
    transit_speed[city] = np.nansum((dataCity.distance_transit / dataCity.duration_transit) * dataCity.density) / np.nansum(dataCity.density)
    
    
df["income"] = income.values()
df["population"] = population.values()
df["pop_growth"] = pop_growth.values()
df["income_growth"] = income_growth.values()
df["beta"] = beta.values()
df["car_speed"] = car_speed.values()
df["transit_speed"] = transit_speed.values()
#data["b"] = b.values()

y = df.iloc[:, 3].values
X1 = df.iloc[:, [4, 5, 6, 7,8]].values
X2 = df.iloc[:, 4:].values


regressor = DecisionTreeClassifier(max_depth = 3, random_state = 0)
regressor.fit(X1, y)
plt.figure(figsize =(22,15))
tree.plot_tree(regressor, feature_names= ["income", "population", "pop growth", "income growth", "beta"], class_names = ['Recommended', 'Small impact', 'Big impact', 'To avoid'], 
              label = 'none', impurity = False, fontsize = 20, proportion = False, rounded = True, precision = 0)

regressor = DecisionTreeClassifier(max_depth = 3, random_state = 0)
regressor.fit(X2, y)
plt.figure(figsize =(22,15))
tree.plot_tree(regressor, feature_names= ["income", "population", "pop growth", "income growth", "beta", "car speed", "transit speed"], class_names = ['Recommended', 'Small impact', 'Big impact', 'To avoid'], 
              label = 'none', impurity = False, fontsize = 20, proportion = False, rounded = True, precision = 0)


#### STEP 3 : GREENBELT

#create the dataset
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    emissions_per_capita = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_greenbelt/" + city + "_emissions_per_capita.npy")
    utility = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_greenbelt/" + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    
greenbelt = pd.DataFrame()
greenbelt["city"] = emission_2015.keys()
greenbelt["emissions_2015"] = emission_2015.values()
greenbelt["emissions_2035"] = emission_2035.values()
greenbelt["utility_2015"] = utility_2015.values()
greenbelt["utility_2035"] = utility_2035.values()

#add continents
city_continent = pd.read_csv("C:/Users/Coupain/Desktop/these/Data/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
greenbelt = greenbelt.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

#compare to baseline
baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
greenbelt = greenbelt.merge(baseline, left_on = "city", right_on = "city", how = 'left')
greenbelt["emissions_reduction_2015"] = greenbelt.emissions_2015 / greenbelt.emissions_2015_BAU
greenbelt["utility_reduction_2015"] = greenbelt.utility_2015 / greenbelt.utility_2015_BAU
greenbelt["emissions_reduction_2035"] = greenbelt.emissions_2035 / greenbelt.emissions_2035_BAU
greenbelt["utility_reduction_2035"] = greenbelt.utility_2035 / greenbelt.utility_2035_BAU

#plot

color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
#plt.rcParams.update({'font.size': 25})
colors = list(greenbelt['Continent'].unique())
plt.figure(figsize=(20, 20))
plt.xlim(-38, 2)
plt.ylim(-75, 0)
#for i in range(0 , len(colors)):
  #  data = greenbelt.loc[greenbelt['Continent'] == colors[i]]
  #  plt.scatter((data.utility_reduction_2035 - 1) * 100, (data.emissions_reduction_2035 - 1) * 100, label=colors[i], s = 200, color=data.Continent.map(color_tab))
plt.scatter((greenbelt.utility_reduction_2035 - 1) * 100, (greenbelt.emissions_reduction_2035 - 1) * 100, s = 200)
#for i in range(192):
#    plt.annotate(carbon_tax.city[i], (carbon_tax.utility_reduction_2035[i], carbon_tax.emissions_reduction_2035[i]))
plt.xlabel("Utility reduction compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.hlines(-30, -38, 2, colors = "black")
plt.vlines(-5, -75, 0, colors = "black")
plt.text(-5, -72, 'Appropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
plt.text(-35, -4, 'Inappropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
#plt.legend()

sum((greenbelt.emissions_reduction_2035 < 0.25) | (greenbelt.utility_reduction_2035 < 0.60))
sum((greenbelt.emissions_reduction_2035 > 1) | (greenbelt.utility_reduction_2035 > 1.02))
#12 outliers !

recommended = greenbelt.city[(greenbelt.utility_reduction_2035 > 0.95) & (greenbelt.emissions_reduction_2035 < 0.7)]
to_avoid = greenbelt.city[(greenbelt.utility_reduction_2035 < 0.95) & (greenbelt.emissions_reduction_2035 > 0.7)]
big_impact = greenbelt.city[(greenbelt.utility_reduction_2035 < 0.95) & (greenbelt.emissions_reduction_2035 < 0.7)]
small_impact = greenbelt.city[(greenbelt.utility_reduction_2035 > 0.95) & (greenbelt.emissions_reduction_2035 > 0.7)]


### regression tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

df = copy.deepcopy(greenbelt.iloc[:, [0,13,14]])
df["type"] = "A"
df.type[(df.utility_reduction_2035 < 0.95) & (df.emissions_reduction_2035 > 0.7)] = "D"
df.type[(df.utility_reduction_2035 > 0.95) & (df.emissions_reduction_2035 > 0.7)] = "B"
df.type[(df.utility_reduction_2035 < 0.95) & (df.emissions_reduction_2035 < 0.7)] = "C"
df = df.sort_values("city")

income = {}
population = {}
pop_growth = {}
income_growth = {}
car_speed = {}
transit_speed = {}

for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    dataCity = pickle.load(open("C:/Users/Coupain/Desktop/these/Sorties/data/" + city + "filename.pickle", "rb", -1))
    income[city] = dataCity.income
    population[city] = dataCity.total_population
    country = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").country[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    region = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").region[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    pop_growth_rate = import_city_scenarios(city, country)
    if isinstance(pop_growth_rate["2015-2020"], pd.Series) == True:
        pop_growth_rate = import_country_scenarios(country)
    pop_growth[city] = ((1+ (pop_growth_rate["2015-2020"]/100)) ** 5) * ((1+(pop_growth_rate["2020-2025"]/100)) ** 5) * ((1+(pop_growth_rate["2025-2030"]/100)) ** 5) * ((1+(pop_growth_rate["2030-2035"]/100)) ** 5)
    imaclim = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx', sheet_name = 'Extraction_Baseline')
    imaclim = imaclim[imaclim.Region == region]
    scenar_income = imaclim[imaclim.Variable == "Index_income"].squeeze()
    income_growth[city] = scenar_income[2035]/scenar_income[2015]
    car_speed[city] = np.nansum((dataCity.distance_driving / dataCity.duration_driving) * dataCity.density) / np.nansum(dataCity.density)
    transit_speed[city] = np.nansum((dataCity.distance_transit / dataCity.duration_transit) * dataCity.density) / np.nansum(dataCity.density)
    
    
df["income"] = income.values()
df["population"] = population.values()
df["pop_growth"] = pop_growth.values()
df["income_growth"] = income_growth.values()
#df["beta"] = beta.values()
df["car_speed"] = car_speed.values()
df["transit_speed"] = transit_speed.values()
#data["b"] = b.values()

y = df.iloc[:, 3].values
X = df.iloc[:, 4:].values


regressor = DecisionTreeClassifier(max_depth = 3, random_state = 0)
regressor.fit(X, y)
plt.figure(figsize = (22, 15))
tree.plot_tree(regressor, feature_names= ["income", "population", "pop growth", "income_growth", "car speed", "transit speed"], class_names = ['Recommended', 'Small impact', 'Big impact', 'To avoid'], 
              label = 'none', impurity = False, fontsize = 20, proportion = False, rounded = True, precision = 0)


#### STEP 4: TRANSIT SPEED

#create the dataset
emission_2015 = {}
emission_2035 = {}
utility_2015 = {}
utility_2035 = {}
for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    emissions_per_capita = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_transit_speed/" + city + "_emissions_per_capita.npy")
    utility = np.load("C:/Users/Coupain/Desktop/these/Sorties/simulation_transit_speed/" + city + "_utility.npy")
    emission_2015[city] = emissions_per_capita[0]
    emission_2035[city] = emissions_per_capita[20]
    utility_2015[city] = utility[0]
    utility_2035[city] = utility[20]
    
transit_speed = pd.DataFrame()
transit_speed["city"] = emission_2015.keys()
transit_speed["emissions_2015"] = emission_2015.values()
transit_speed["emissions_2035"] = emission_2035.values()
transit_speed["utility_2015"] = utility_2015.values()
transit_speed["utility_2035"] = utility_2035.values()

#add continents
city_continent = pd.read_csv("C:/Users/Coupain/Desktop/these/Data/cityDatabase_221NewCities.csv")
city_continent = city_continent.iloc[:, [0, 2]]
city_continent = city_continent.drop_duplicates(subset = "City")
city_continent = city_continent.sort_values('City')
transit_speed = transit_speed.merge(city_continent, left_on = "city", right_on = "City", how = 'left')

#compare to baseline
baseline = copy.deepcopy(no_policy.iloc[:, [0, 1, 2, 3, 4]])
baseline.columns = ['city', 'emissions_2015_BAU', 'emissions_2035_BAU', 'utility_2015_BAU', 'utility_2035_BAU']
transit_speed = transit_speed.merge(baseline, left_on = "city", right_on = "city", how = 'left')
transit_speed["emissions_reduction_2015"] = transit_speed.emissions_2015 / transit_speed.emissions_2015_BAU
transit_speed["utility_reduction_2015"] = transit_speed.utility_2015 / transit_speed.utility_2015_BAU
transit_speed["emissions_reduction_2035"] = transit_speed.emissions_2035 / transit_speed.emissions_2035_BAU
transit_speed["utility_reduction_2035"] = transit_speed.utility_2035 / transit_speed.utility_2035_BAU

#plot

#color_tab = {'North_America':'red', 'Europe':'green', 'Asia':'blue', 'Oceania':'yellow', 'South_America': 'brown', 'Africa': 'orange'}
#plt.rcParams.update({'font.size': 25})
#colors = list(transit_speed['Continent'].unique())
plt.figure(figsize=(20, 20))
plt.xlim(0, 1.5)
plt.ylim(-53, 1)
#for i in range(0 , len(colors)):
    #☺data = transit_speed.loc[transit_speed['Continent'] == colors[i]]
plt.scatter((transit_speed.utility_reduction_2035 - 1) * 100, (transit_speed.emissions_reduction_2035 - 1) * 100,s = 200)
#for i in range(192):
#    plt.annotate(carbon_tax.city[i], (carbon_tax.utility_reduction_2035[i], carbon_tax.emissions_reduction_2035[i]))
plt.xlabel("Utility gain compared to BAU (%)")
plt.ylabel("Emissions per capita reduction compared to BAU (%)")
plt.hlines(-10, -0.1, 1.5, colors = "black")
plt.vlines(0.2, -55, 1, colors = "black")
plt.text(1.2, -50, 'Appropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
plt.text(0.1, -2, 'Inappropriate', color='red', 
        bbox=dict(facecolor='none', edgecolor='red', boxstyle='round,pad=1'))
#plt.legend()

#2 outliers !

recommended = transit_speed.city[(transit_speed.utility_reduction_2035 > 1.002) & (transit_speed.emissions_reduction_2035 < 0.9)]
to_avoid = transit_speed.city[(transit_speed.utility_reduction_2035 < 1.002) & (transit_speed.emissions_reduction_2035 > 0.9)]
utility_gain = transit_speed.city[(transit_speed.utility_reduction_2035 > 1.002) & (transit_speed.emissions_reduction_2035 > 0.9)]
emissions_decrease = transit_speed.city[(transit_speed.utility_reduction_2035 < 1.002) & (transit_speed.emissions_reduction_2035 < 0.9)]


### regression tree

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

df = copy.deepcopy(transit_speed.iloc[:, [0,13,14]])
df["type"] = "A"
df.type[(df.utility_reduction_2035 < 1.002) & (df.emissions_reduction_2035 > 0.9)] = "D"
df.type[(df.utility_reduction_2035 < 1.002) & (df.emissions_reduction_2035 < 0.9)] = "B" #emissions_decreaseNo
df.type[(df.utility_reduction_2035 > 1.002) & (df.emissions_reduction_2035 > 0.9)] = "C" #utility_gain
df = df.sort_values("city")

income = {}
population = {}
pop_growth = {}
income_growth = {}
car_speed = {}
transit_speed = {}

for index in range(0, 192):
    city = arr[index * 2].replace('filename.pickle', '')
    dataCity = pickle.load(open("C:/Users/Coupain/Desktop/these/Sorties/data/" + city + "filename.pickle", "rb", -1))
    income[city] = dataCity.income
    population[city] = dataCity.total_population
    country = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").country[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    region = pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").region[pd.read_excel("C:/Users/Coupain/Desktop/these/Data/city_country_region.xlsx").city == city].squeeze()
    pop_growth_rate = import_city_scenarios(city, country)
    if isinstance(pop_growth_rate["2015-2020"], pd.Series) == True:
        pop_growth_rate = import_country_scenarios(country)
    pop_growth[city] = ((1+ (pop_growth_rate["2015-2020"]/100)) ** 5) * ((1+(pop_growth_rate["2020-2025"]/100)) ** 5) * ((1+(pop_growth_rate["2025-2030"]/100)) ** 5) * ((1+(pop_growth_rate["2030-2035"]/100)) ** 5)
    imaclim = pd.read_excel('C:/Users/Coupain/Desktop/these/Data/Charlotte_ResultsIncomePriceAutoEmissionsAuto_V1.xlsx', sheet_name = 'Extraction_Baseline')
    imaclim = imaclim[imaclim.Region == region]
    scenar_income = imaclim[imaclim.Variable == "Index_income"].squeeze()
    income_growth[city] = scenar_income[2035]/scenar_income[2015]
    car_speed[city] = np.nansum((dataCity.distance_driving / dataCity.duration_driving) * dataCity.density) / np.nansum(dataCity.density)
    transit_speed[city] = np.nansum((dataCity.distance_transit / dataCity.duration_transit) * dataCity.density) / np.nansum(dataCity.density)
    
    
df["income"] = income.values()
df["population"] = population.values()
df["pop_growth"] = pop_growth.values()
#df["income_growth"] = income_growth.values()
df["beta"] = beta.values()
df["car_speed"] = car_speed.values()
df["transit_speed"] = transit_speed.values()
#data["b"] = b.values()

y = df.iloc[:, 3].values
X1 = df.iloc[:, [4, 5, 6, 7]].values
X2 = df.iloc[:, 4:].values


regressor = DecisionTreeClassifier(max_depth = 3, random_state = 0)
regressor.fit(X1, y)
plt.figure(figsize =(22,15))
tree.plot_tree(regressor, feature_names= ["income", "population", "pop growth", "beta"], class_names = ['Recommended', 'Emissions decrease', 'Utility gain', 'To avoid'], 
              label = 'none', impurity = False, fontsize = 20, proportion = False, rounded = True, precision = 0)

regressor = DecisionTreeClassifier(max_depth = 3, random_state = 0)
regressor.fit(X2, y)
plt.figure(figsize =(22,15))
tree.plot_tree(regressor, feature_names= ["income", "population", "pop growth", "beta", "car speed", "transit speed"], class_names = ['Recommended', 'Emissions decrease', 'Utility gain', 'To avoid'], 
              label = 'none', impurity = False, fontsize = 20, proportion = False, rounded = True, precision = 0)
