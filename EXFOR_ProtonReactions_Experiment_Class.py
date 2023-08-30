"""
================================================================================
TITLE: Experiment Data Processing and Visualization
AUTHOR: Juan A. Monleón de la Lluvia
DATE: 28-08-2023

DESCRIPTION:
This script is designed for managing scientific experiments and their associated data. 
It includes a class, `Experiment`, with methods for data manipulation and visualization. 
This includes attribute initialization, data transformation to Pandas DataFrame, 
numerical and categorical attribute encoding, as well as plotting capabilities.

MAIN FEATURES:
- Representation of experimental setup t

hrough object attributes.
- Conversion of experiment attributes to Pandas DataFrame for further analysis.
- Addition of numerical attributes to the DataFrame.
- One-hot encoding of categorical attributes for model training.
- Data preparation that includes both numerical conversion and one-hot encoding.
- Plotting capabilities with optional logarithmic scaling.

DEPENDENCIES:
- pandas
- numpy
- matplotlib
- seaborn

USAGE:
1. Initialize an Experiment object.
2. Assign values to experiment attributes.
3. Call relevant methods for data manipulation or visualization.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Experiment:
    """
    This class models an experimental setup for scientific data collection.
    Attributes include various experimental parameters and data in the form of a Pandas DataFrame.
    """
    def __init__(self):
        """
        Initializes an Experiment object with default values set to None and an empty Pandas DataFrame for data.
        """
        self.title = None
        self.target_Z = None
        self.target_A = None
        self.target_state = None
        self.projectile = None
        self.reaction = None
        self.E_inc = None
        self.final_Z = None
        self.final_A = None
        self.final_state = None
        self.MTrat = None
        self.Ratio_isomer = None
        self.quantity = None
        self.frame = None
        self.MF = None
        self.MT = None
        self.X4_ID = None
        self.X4_code = None
        self.author = None
        self.year = None
        self.data_points = None
        self.data = pd.DataFrame()
        self.reference = None


    def __str__(self):
        """
        Overloaded string method to print non-empty attributes of an Experiment object.
        """
        print('Title: ', self.title)
        if self.target_Z != None: print('Target Z: ', self.target_Z)
        if self.target_A != None: print('Target A: ', self.target_A)
        if self.target_state != None: print('Target state: ', self.target_state)
        if self.projectile != None: print('Projectile: ', self.projectile)
        if self.reaction != None: print('Reaction: ', self.reaction)
        if self.E_inc != None: print('Incident energy: ', self.E_inc)
        if self.final_Z != None: print('Final Z: ', self.final_Z)
        if self.final_A != None: print('Final A: ', self.final_A)
        if self.final_state != None: print('Final state: ', self.final_state)
        if self.MTrat != None: print('MT ratio: ', self.MTrat)
        if self.Ratio_isomer != None: print('Ratio isomer: ', self.Ratio_isomer)
        if self.quantity != None: print('Quantity: ', self.quantity)
        if self.frame != None: print('Frame: ', self.frame)
        if self.MF != None: print('MF: ', self.MF)
        if self.MT != None: print('MT: ', self.MT)
        if self.X4_ID != None: print('X4 ID: ', self.X4_ID)
        if self.X4_code != None: print('X4 code: ', self.X4_code)
        if self.author != None: print('Author: ', self.author)
        if self.year != None: print('Year: ', self.year)
        if self.data_points != None: print('Data points: ', self.data_points)
        if not self.data.empty: print('Data: ', self.data)
        if self.reference != None: print('Reference:\n', self.reference)
        return ''


    def to_dataframe(self):
        """
        Converts the attributes of the object into a single DataFrame where each attribute is a column.
        Returns a DataFrame containing the enriched data.
        """
        # Starting with the data in self.data, we are going to add information from the other attributes
        # The information will be added as a new column in the dataframe for all the rows in the dataframe
        
        # List with all the attributes to add to the dataframe
        attributes = ['Experiment', 'Target Z', 'Target A', 'Target state', 'Reaction', 'Incident energy', 
                      'Final Z', 'Final A', 'Final state', 'MT ratio', 'Ratio isomer', 'Quantity', 'Frame', 
                      'MF', 'MT', 'Author', 'Year']

        # Create a copy of the data
        data = self.data.copy()
        # Add the attributes to the dataframe
        for attribute in attributes:
            # Create a list with the value of the attribute for all the rows in the dataframe
            if attribute == 'Experiment': value = [self.title]*len(data)
            elif attribute == 'Target Z': value = [self.target_Z]*len(data)
            elif attribute == 'Target A': value = [self.target_A]*len(data)
            elif attribute == 'Target state': value = [self.target_state]*len(data)
            elif attribute == 'Reaction': value = [self.reaction]*len(data)
            elif attribute == 'Incident energy': value = [self.E_inc]*len(data)
            elif attribute == 'Final Z': value = [self.final_Z]*len(data)
            elif attribute == 'Final A': value = [self.final_A]*len(data)
            elif attribute == 'Final state': value = [self.final_state]*len(data)
            elif attribute == 'MT ratio': value = [self.MTrat]*len(data)
            elif attribute == 'Ratio isomer': value = [self.Ratio_isomer]*len(data)
            elif attribute == 'Quantity': value = [self.quantity]*len(data)
            elif attribute == 'Frame': value = [self.frame]*len(data)
            elif attribute == 'MF': value = [self.MF]*len(data)
            elif attribute == 'MT': value = [self.MT]*len(data)
            elif attribute == 'Author': value = [self.author]*len(data)
            elif attribute == 'Year': value = [self.year]*len(data)
            # Add the column to the dataframe
            data[attribute] = value
        # Return the dataframe
        return data
    

    def add_numeric_attributes(self):
        """
        Adds numeric attributes to the DataFrame by converting suitable string values to their respective numeric forms.
        """
        attributes = ['E_inc', 'MF', 'MT', 'MTrat', 'Ratio_isomer', 'final_A', 'final_Z', 'target_A', 'target_Z']

        for attr in attributes:
            if attr == 'E_inc':
                value = getattr(self, attr)
                if value is not None:
                    first_element = value.split()[0]
                    self.data[attr] = float(first_element)
                else:
                    self.data[attr] = None
            elif attr in ['MTrat', 'final_A', 'final_Z']:
                value = getattr(self, attr)
                if value is not None:
                    self.data[attr] = int(value)
                else:
                    self.data[attr] = None
            else:
                self.data[attr] = getattr(self, attr)


    def encode_categorical_attributes(self):
        """
        Transforms categorical attributes to a format suitable for model training by using one-hot encoding.
        """
        categorical_attributes = {
            'projectile': ['projectile_p'],
            'final_state': ['final_state_+', 'final_state_1', 'final_state_2', 'final_state_G', 'final_state_M'],
            'frame': ['frame_C', 'frame_L'],
            'quantity': ['qty_Angular distribution', 'qty_Cross section', 'qty_Cross section ratio', 'qty_Delayed nubar', 'qty_Differential cross section', 'qty_Fission yields', 'qty_Prompt nubar', 'qty_Resonance Parameters', 'qty_Total nubar'],
            'reaction': ['reaction_(n, )', 'reaction_(p, el)', 'reaction_(p, f)', 'reaction_(p, x)', 'reaction_(p, xa)', 'reaction_(p, xd)', 'reaction_(p, xg)', 'reaction_(p, xh)', 'reaction_(p, xn)', 'reaction_(p, xp)', 'reaction_(p, xt)', 'reaction_(p,2a)', 'reaction_(p,2n)', 'reaction_(p,2n)g', 'reaction_(p,2n)m', 'reaction_(p,2na)', 'reaction_(p,2np)', 'reaction_(p,2p)', 'reaction_(p,2p)g', 'reaction_(p,2p)m', 'reaction_(p,3a)', 'reaction_(p,3n)', 'reaction_(p,3n)g', 'reaction_(p,3n)m', 'reaction_(p,3n)n', 'reaction_(p,3na)', 'reaction_(p,3np)', 'reaction_(p,3np)g', 'reaction_(p,3np)m', 'reaction_(p,4n)', 'reaction_(p,4n)g', 'reaction_(p,4n)m', 'reaction_(p,a)', 'reaction_(p,a)g', 'reaction_(p,a)m', 'reaction_(p,d)', 'reaction_(p,d2a)', 'reaction_(p,da)', 'reaction_(p,f)', 'reaction_(p,f)g', 'reaction_(p,f)m', 'reaction_(p,f)n', 'reaction_(p,g)', 'reaction_(p,g)g', 'reaction_(p,g)m', 'reaction_(p,h)', 'reaction_(p,h)g', "reaction_(p,n')", "reaction_(p,n')g", "reaction_(p,n')m", "reaction_(p,n')n", "reaction_(p,n'_01)", "reaction_(p,n'_40)", 'reaction_(p,n2a)', 'reaction_(p,n2p)', 'reaction_(p,n3a)', 'reaction_(p,na)', 'reaction_(p,na)g', 'reaction_(p,na)m', 'reaction_(p,non)', 'reaction_(p,np)', 'reaction_(p,np)g', 'reaction_(p,np)m', 'reaction_(p,npa)', 'reaction_(p,p)', 'reaction_(p,p)m', 'reaction_(p,pa)', 'reaction_(p,pd)', 'reaction_(p,pt)', 'reaction_(p,t)', 'reaction_(p,xa)', 'reaction_(p,xd)', 'reaction_(p,xg)', 'reaction_(p,xh)', 'reaction_(p,xn)', 'reaction_(p,xp)', 'reaction_(p,xt)', 'reaction_Exchange_scattering', 'reaction_Inelastic_scattering', 'reaction_ratio'],
            'target_state': ['target_state_m']
        }
        new_columns_data = {}

        for attribute, possible_values in categorical_attributes.items():
            # Get the current attribute value of this instance
            current_attribute_value = getattr(self, attribute)
            
            for value in possible_values:
                # Get the value after the last underscore
                comparison_value = value.split('_')[-1]
                # Check if the attribute value of the current instance matches the comparison value
                new_columns_data[value] = int(current_attribute_value == comparison_value)

        # Convert the dictionary into a DataFrame and concatenate it with the original DataFrame
        new_columns_df = pd.DataFrame(new_columns_data, index=self.data.index)
        self.data = pd.concat([self.data, new_columns_df], axis=1)



    def prepare_data(self):
        """
        Combines the functionalities of add_numeric_attributes and encode_categorical_attributes to prepare the data for analysis.
        Returns the prepared DataFrame.
        """
        self.add_numeric_attributes()
        self.encode_categorical_attributes()
        self.data['X4_ID'] = str(getattr(self, 'X4_ID'))
        return self.data


    def plot(self, xlog=False, ylog=False, fig_size=(9,6)):
        """
        Creates a plot of the stored data with optional logarithmic scaling for x and/or y axes.
        Parameters:
        - xlog (bool): Whether to use log scale on the x-axis.
        - ylog (bool): Whether to use log scale on the y-axis.
        - fig_size (tuple): Tuple specifying the dimensions of the plot.
        """
        if self.data.empty:
            print('No data to plot')
        else:
            # check if the data in the third column are all NaN or 0
            y_err = self.data.iloc[:,2].isnull().values.all() or self.data.iloc[:,2].eq(0).all()
            # check if the data in the fourth column are all NaN or 0
            x_err = self.data.iloc[:,3].isnull().values.all() or self.data.iloc[:,3].eq(0).all()
            
            # Get the headers of the data
            headers = list(self.data.columns.values)
        
            # Set the size of the plot
            plt.figure(figsize=fig_size)
            # Plot the data with a scatter plot
            # If y_err and x_err are False, plot the data with error bars
            if y_err == False and x_err == False:
                plt.errorbar(x=self.data[headers[0]], y=self.data[headers[1]], 
                            xerr=self.data[headers[3]], yerr=self.data[headers[2]],
                            fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5)
            # If y_err is False, plot the data with error bars on the y-axis
            elif y_err == False:
                plt.errorbar(x=self.data[headers[0]], y=self.data[headers[1]], yerr=self.data[headers[2]], 
                            fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5,)
            # If x_err is False, plot the data with error bars on the x-axis
            elif x_err == False:
                plt.errorbar(x=self.data[headers[0]], y=self.data[headers[1]], xerr=self.data[headers[3]], 
                            fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5,)
            # If y_err and x_err are False, plot the data without error bars
            else:
                sns.scatterplot(x=headers[0], y=headers[1], data=self.data)


            # Set the scale of the plot
            if xlog == True: plt.xscale('log')
            if ylog == True: plt.yscale('log')

            # Set the title of the plot
            plt.title(self.title, fontsize=18, pad=15)
            # Set the labels of the plot
            plt.xlabel(headers[0], fontsize=16, labelpad=10)
            plt.ylabel(headers[1], fontsize=16, labelpad=10)
            # Set the ticks of the plot
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            # Show the plot
            plt.show()
            


