# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:32:54 2021

@author: Coupain
"""

import numpy as np
import pandas as pd

class CityClass:  
    """Classe définissant une ville caractérisée par :
        - les densités en chaque pixel
        - les loyers en chaque pixel
        - les tailles des logements en chaque pixel
        - la proportion de chaque pixel qui est urbanisable
        - le revenu moyen à l échelle de la ville
        - les temps de transport au centre depuis chaque pixel
        """
        
    def __init__(self,
                 density=0,
                 rent=0,
                 size=0,
                 urb=0,
                 income=0,
                 duration=0,
                 prices=0,
                 duration_driving=0,
                 distance_driving=0,
                 duration_transit=0,
                 distance_transit=0,
                 mode_choice=0
                 ):
        self.density = density
        self.rent = rent
        self.size = size
        self.urb = urb
        self.income = income
        self.duration = duration
        self.prices=prices #real estate prices
        self.duration_driving=duration_driving
        self.distance_driving=distance_driving
        self.duration_transit=duration_transit
        self.distance_transit=distance_transit
        self.mode_choice=mode_choice
        
    
    def drop_outliers(self, PARAM_OUTLIERS):
        size = self.size
        rent = self.rent
        limit_size = np.mean(size) + (PARAM_OUTLIERS * np.std(size))
        size.loc[size > limit_size]= np.nan
        
        limit_rent = np.mean(rent) + (PARAM_OUTLIERS * np.std(rent))
        rent.loc[rent> limit_rent]= np.nan
        self.size = size
        self.rent = rent
    
    def environmental_outputs(self, grid, beta, urban_footprint_residual):
        """ Compute the environmental outputs:
            - Urban footprint (from density)
            - Average distance to CBD
            - Total distance to CBD """
            
        self.urban_footprint = np.minimum(((self.density * beta) + urban_footprint_residual), np.ones(len(self.density)))
        self.urban_footprint = np.maximum(self.urban_footprint, 0)
        indices = np.where(np.logical_not(np.isnan(self.density)))[0]
        self.average_distance_to_cbd = np.ma.average(grid.distance_cbd[indices], weights = self.density[indices])
        self.total_distance_to_cbd = np.nansum(grid.distance_cbd * self.density)
        self.total_distance_to_cbd_car = np.nansum(grid.distance_cbd * self.density * (self.mode_choice==0))
        
    def __repr__(self):
        """Quand on entre notre objet dans l'interpréteur."""
        return "density:\n  rent: {}\n  size: {}\n  size: {}\n  urb: {}\n income: {}\n duration: {}".format(
                self.density, 
                self.rent, 
                self.size,
                self.urb,
                self.income,
                self.duration)
    
    def give_total_population(self):
        total_population=np.nansum(self.density)
        return total_population
        
    def replicate(self,orig):
        self.density = orig.density
        self.rent = orig.rent
        self.size = orig.size
        self.urb = orig.urb
        self.income = orig.income
        self.duration = orig.duration
        self.prices=orig.prices #real estate prices
        self.duration_driving=orig.duration_driving
        self.distance_driving=orig.distance_driving
        self.duration_transit=orig.duration_transit
        self.distance_transit=orig.distance_transit
        self.mode_choice=orig.mode_choice
        

class Grid:
    """Classe définissant une grille caractérisée par :
    - coord_X
    - coord_Y
    - distance_centre
    - area"""
    
    def __init__(self,coord_X=0,coord_Y=0,distance_cbd=0,area=0): # Notre méthode constructeur
        """Constructeur de la classe"""
        self.coord_X = coord_X
        self.coord_Y = coord_Y
        self.distance_cbd = distance_cbd
        self.area = area
        
        
    def create_grid(self,n):
        """Cree une grille de n*n pixels, centree en 0"""
        coord_X = np.zeros(n*n)
        coord_Y = np.zeros(n*n)
    
        indexu = 0
        for i in range(n) :
            for j in range(n) :
                coord_X[indexu] = i - n / 2
                coord_Y[indexu] = j - n / 2
                indexu = indexu + 1
        distance_cbd = (coord_X ** 2 + coord_Y ** 2) ** 0.5
    
        self.coord_X = coord_X
        self.coord_Y = coord_Y
        self.distance_cbd = distance_cbd
        self.area = 1

    def __repr__(self):
        """Quand on entre notre objet dans l'interpréteur."""
        return "Grid:\n  coord_X: {}\n  coord_Y: {}\n  distance_cbd: {}\n  area: {}".format(
                self.coord_X, self.coord_Y, self.distance_cbd,self.area) 
   
class TransportSimulation:
    
    def __init__(self,price_car=0,price_public_transport=0,mode=0,transport_price=0):
        self.price_car = price_car
        self.price_public_transport = price_public_transport
        self.mode = mode
        self.transport_price = transport_price
        
    def create_trans(self, dataCity, policy, index, scenar_driving_price = None, initial_year = None):
        
        if isinstance(scenar_driving_price, pd.Series) == False:
            scenar_driving_price = pd.Series(np.repeat(1, 49), index = np.arange(2002, 2051)) 
            initial_year = 2015
        if policy["carbon_tax_implementation"][index] == 0:
            prix_driving = dataCity.duration_driving * dataCity.income / (3600 * 24) / 365 + ((dataCity.distance_driving * 0.860 * 7.18 / 100000) * (scenar_driving_price[initial_year + index] / scenar_driving_price[initial_year]))
        elif policy["carbon_tax_implementation"][index] == 1:
            prix_driving = dataCity.duration_driving * dataCity.income / (3600 * 24) / 365 + (dataCity.distance_driving * (policy["carbon_tax_value"]) * 7.18 / 100000) + ((dataCity.distance_driving * 0.860 * 7.18 / 100000) * (scenar_driving_price[initial_year + index] / scenar_driving_price[initial_year]))
        if policy["public_transport_speed_implementation"][index] == 0:
            prix_transit = dataCity.duration_transit * dataCity.income / (3600 * 24) / 365 + ((dataCity.distance_transit * 0.860 * 7.18 / 100000)  * (scenar_driving_price[initial_year + index] / scenar_driving_price[initial_year]))
        elif policy["public_transport_speed_implementation"][index] == 1:
            prix_transit = (dataCity.duration_transit / policy["public_transport_speed_factor"]) * dataCity.income / (3600 * 24) / 365 + ((dataCity.distance_transit * 0.860 * 7.18 / 100000)  * (scenar_driving_price[initial_year + index] / scenar_driving_price[initial_year]))
            
        tous_prix=np.vstack((prix_driving,prix_transit))#les prix concaténés
        prix_transport=np.amin(tous_prix, axis=0)
        prix_transport[np.isnan(prix_transit)]=prix_driving[np.isnan(prix_transit)]
        prix_transport=prix_transport*2*365 # on l'exprime par rapport au revenu
        prix_transport=pd.Series(prix_transport)
        mode_choice=np.argmin(tous_prix, axis=0)
        mode_choice[np.isnan(prix_transit)]=0
        
        self.price_car = prix_driving
        self.price_public_transport = prix_transit
        self.mode = mode_choice
        self.transport_price = prix_transport

    def __repr__(self):
        return ("Trans:\nprice_car: {}\nprice_public_transport: {}\nmode: {}\ntransport_price: {}\n".format(
            self.price_car,self.price_public_transport,self.mode,self.transport_price))

class Residuals:  
    """ Classe définissant les résidus à l'issue de la calibration """
    
    def __init__(self,
                 density_residual=0,
                 rent_residual=0,
                 size_residual=0):
        self.density_residual = density_residual
        self.rent_residual = rent_residual
        self.size_residual = size_residual
