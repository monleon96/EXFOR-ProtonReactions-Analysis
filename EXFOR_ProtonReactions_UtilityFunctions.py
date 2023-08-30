"""
================================================================================
TITLE: Utility Functions for Proton Experiment Data Analysis
AUTHOR: Juan A. Monleón de la Lluvia
DATE: 28-08-2023

DESCRIPTION: 
    This Python script provides a collection of functions and classes aimed at
    processing and visualizing data from proton experiments. It includes methods
    for reading experiment data, performing statistical analyses, and generating 
    plots for better understanding of the results.

MAIN FEATURES:
    - Data Reading: Reads experiment data from specified file formats.
    - Data Manipulation: Utilizes pandas and numpy for data cleaning and transformation.
    - Visualization: Uses matplotlib and seaborn for data visualization.
    - Experiment Object: Utilizes a custom Experiment class for better data management.

DEPENDENCIES:
    - pandas
    - numpy
    - matplotlib
    - seaborn
    - Experiment (custom class)

USAGE:
    1. Import the script: `import proton_func as pf`
    2. Read an experiment: `exp = pf.read_experiment('filename')`
    3. Perform analysis: `pf.some_analysis_function(exp)`
    4. Generate plots: `pf.some_plotting_function(exp)`

================================================================================
"""

import pandas as pd
import numpy as np
from EXFOR_ProtonReactions_Experiment_Class import Experiment
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


def read_experiment(filename):
    """
    This function reads a file specified by 'filename', processes its content line by line, and populates the properties 
    of an Experiment object according to specific keywords present in the file. It finally returns this populated 
    Experiment object.

    File Format:
    The file should contain lines starting with specific keywords, followed by a colon and the value (EXFORTABLES format).
    (e.g., "# Target Z: 12")

    Properties of Returned Experiment Object:
    - title (str): Extracted from the filename.
    - target_Z (int or None): Atomic number of the target.
    - target_A (int or None): Mass number of the target.
    - target_state (int/str): the state of the target.
    - projectile (str): the projectile used in the experiment.
    - reaction (str): the reaction that occurred during the experiment.
    - E_inc (str): the incident energy of the projectile.
    - final_Z (int/str): the atomic number of the final state.
    - final_A (int/str): the mass number of the final state.
    - final_state (int/str): the state of the final state.
    - MTrat (float/str): the ratio of the number of gamma rays to the number of neutrons.
    - Ratio_isomer (float/str): the ratio of the number of isomers to the number of ground state.
    - Quantity (str): the quantity of the experiment.
    - Frame (str): the frame of the experiment.
    - MF (int/str): the MF of the experiment.
    - MT (int/str): the MT of the experiment.
    - X4_ID (str): the EXFOR ID of the experiment.
    - X4_code (str): the EXFOR code of the experiment.
    - author (str): the author of the experiment.
    - year (int/str): the year of the experiment.
    - data_points (int/str): the number of data points in the experiment.
    - data (pd.DataFrame): the data of the experiment.
    - Reference (str): the reference of the experiment.

    If a property value is not found in the file or cannot be converted to its respective type, the value is set to None.

    Dependencies:
    - Pandas: Required for storing experimental data in a DataFrame.
    - NumPy: Required for handling 'nan' values.

    Parameters:
        filename (str): Name or path of the file to read.

    Returns:
        experiment (Experiment): Populated Experiment object.

    Raises:
        FileNotFoundError: If the file specified by 'filename' does not exist.
    """

    # Create the experiment
    experiment = Experiment()
    experiment.title = filename.split('\\')[-1]
    print('Reading experiment: ', experiment.title)
    read_header = False         # Flag to read the header
    read_ref = False            # Flag to read the reference
    temp_line = ""              # Temporary line to store the reference

    # Open the file
    with open(filename, 'r') as f:
        # Read the first line
        lines = f.readlines()
        for line in lines:

            if line.startswith('#') and not read_ref and not read_header:            
                if line.startswith('# Target Z'):
                    try:
                        experiment.target_Z = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.target_Z = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Target A'):
                    try:
                        experiment.target_A = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.target_A = int(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Target state'):
                    try:
                        experiment.target_state = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.target_state = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Projectile'):
                    experiment.projectile = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Reaction    :'):           # This includes the ':' because there is another line with the same beginning in some files
                    experiment.reaction = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# E-inc'):
                    experiment.E_inc = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Final Z'):
                    try:
                        experiment.final_Z = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.final_Z = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Final A'):
                    try:
                        experiment.final_A = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.final_A = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Final state'):
                    try:
                        experiment.final_state = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.final_state = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# MTrat'):
                    try:
                        experiment.MTrat = float(line.split(':')[1].strip())
                    except ValueError:
                        experiment.MTrat = float(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Ratio isomer'):
                    try:
                        experiment.Ratio_isomer = float(line.split(':')[1].strip())
                    except ValueError:
                        experiment.Ratio_isomer = float(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Quantity'):
                    experiment.quantity = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Frame'):
                    experiment.frame = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# MF'):
                    try:
                        experiment.MF = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.MF = int(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# MT'):
                    try:
                        experiment.MT = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.MT = int(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# X4 ID'):
                    experiment.X4_ID = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# X4 code'):
                    experiment.X4_code = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Author'):
                    experiment.author = line.split(':')[1].strip() if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Year'):
                    try:
                        experiment.year = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.year = int(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                elif line.startswith('# Data points'):
                    try:
                        experiment.data_points = int(line.split(':')[1].strip())
                    except ValueError:
                        experiment.data_points = int(line.split(':')[1].strip()) if line.split(':')[1].strip() != '' else None
                    read_header = True           
                elif line.startswith('# Reference'):
                    read_ref = True

            elif read_header:
                header = line.split()[1:]
                data_list = [[] for i in range(len(header))]
                read_header = False

            elif read_ref:
                # If line only contains '#\n', then it is the end of the reference
                if line == '#\n':
                    # add the reference to the experiment except for the last two characters (which are '# ')
                    experiment.reference = temp_line[:-2]
                    read_ref = False        # Reset the flag
                    temp_line = ""          # Reset the temporary line
                    continue                
                temp_line += line[1:]
                
            else:
                data = line.split()
                # Get number of splits
                n = len(data)
                data_list[0].append(float(data[0]))
                data_list[1].append(float(data[1]))
                # If there are 2 splits, then there are no values for dE and dxs
                if n == 2:
                    data_list[2].append(np.nan)
                    data_list[3].append(np.nan)
                # If there are 3 splits, then there are no values for dE
                elif n == 3:
                    data_list[2].append(float(data[2]))
                    data_list[3].append(np.nan)
                # If there are 4 splits, then there are values for dE and dxs
                elif n == 4:
                    data_list[2].append(float(data[2]))
                    data_list[3].append(float(data[3]))  

        if read_ref:        # This being True means that the file ended directly after the reference
            experiment.reference = temp_line[:-2]

        # Close the file
        f.close()
    
    # Create a data frame using 'header' and 'E', 'xs', 'dxs' and 'dE' lists
    experiment.data = pd.DataFrame(data_list, index=header).T

    # Return the experiment
    return experiment


def write_experiments_to_binary(experiments, filename):
    """
    This function writes a list of Experiment objects to a binary file.
    It uses Python's pickle library for object serialization.
    
    Parameters:
        experiments (list): A list of Experiment objects to be written to the file.
        filename (str): The name of the file where the experiments will be written.

    Returns:
        None

    Example:
        experiments = [experiment1, experiment2, experiment3]
        filename = "experiments.bin"
        write_experiments_to_binary(experiments, filename)

    Note:
        The function uses Python's pickle library, so it's essential to ensure that the Experiment objects are picklable.
        Be aware of the potential security risks if you're unpickling data from an untrusted source.
    """
    
    with open(filename, 'wb') as f:
        for experiment in experiments:
            pickle.dump(experiment, f)



def read_experiments_from_binary(filename):
    """
    This function reads a list of Experiment objects from a binary file and returns them as a list.
    It uses Python's pickle library for object deserialization.

    Parameters:
        filename (str): The name of the binary file to read the experiments from.

    Returns:
        loaded_experiments (list): A list of experiments that were read from the binary file.

    Note:
        The function uses Python's pickle library, so be cautious of potential security risks if you're unpickling data from an untrusted source.
    """

    loaded_experiments = []

    # Use a with statement to ensure the file is properly closed after reading
    with open(filename, 'rb') as f:
        while True:
            try:
                loaded_experiments.append(pickle.load(f))
            except EOFError:
                break
            except pickle.PickleError:
                print("Error in deserializing object. Skipping...")
                break

    return loaded_experiments


def write_experiments_to_txt(experiments, filename):
    """
    write_experiments_to_txt(experiments, filename)

    Serializes a list of experiment objects into a text file, capturing various attributes of each experiment. 
    The function handles empty string attributes by writing them as such in the output file. Once all experiment details are written, the file is closed.

    Parameters:
        experiments (list): A list of experiment objects.
                            
        filename (str): The name of the output text file where the experiment details will be serialized.

    Returns:
        None

    Example:
        experiments = [experiment1, experiment2]
        filename = "experiments.txt"
        write_experiments_to_txt(experiments, filename)

    Note:
        Ensure that each experiment object in the list has all the mentioned attributes. Missing attributes will lead to an error.
    """

    with open(filename, 'w') as f:
        for experiment in experiments:
            # Write the header
            f.write('# Title       : ' + experiment.title + '\n')
            # Write the information. If the information is '', then write an empty string
            f.write('# Reaction    : ' + experiment.reaction  + '\n' if experiment.reaction is not None else '# Reaction    : \n')
            f.write('# Ratio isomer: ' + str(experiment.Ratio_isomer) + '\n' if experiment.Ratio_isomer is not None else '# Ratio isomer: \n')
            f.write('# Quantity    : ' + experiment.quantity + '\n' if experiment.quantity is not None else '# Quantity    : \n')
            f.write('# Frame       : ' + experiment.frame + '\n' if experiment.frame is not None else '# Frame       : \n')
            f.write('# MF          : ' + str(experiment.MF) + '\n' if experiment.MF is not None else '# MF          : \n')
            f.write('# MT          : ' + str(experiment.MT) + '\n' if experiment.MT is not None else '# MT          : \n')
            f.write('# X4 ID       : ' + experiment.X4_ID + '\n' if experiment.X4_ID is not None else '# X4 ID       : \n')
            f.write('# X4 code     : ' + experiment.X4_code + '\n' if experiment.X4_code is not None else '# X4 code     : \n')
            f.write('# Author      : ' + experiment.author + '\n' if experiment.author is not None else '# Author      : \n')
            f.write('# Year        : ' + str(experiment.year) + '\n' if experiment.year is not None else '# Year        : \n')
            f.write('# Data points : ' + str(experiment.data_points) + '\n' if experiment.data_points is not None else '# Data points : \n')    
            # Write the data
            f.write(experiment.data.to_string(header=True, index=False))
            # Write the reference 
            # If the reference is '', then write an empty string
            f.write('\n# Reference   : \n' + experiment.reference + '\n' if experiment.reference is not None else '# Reference   : \n') 
            # Write a line to separate the reference from the next experiment
            f.write('# END\n')
    
        f.write('# END OF FILE')


def read_experiments_from_txt(filename):
    """
    Reads a text file containing a structured list of experiments and returns a list of Experiment objects.

    The function expects the file to be structured as follows:
    - Each experiment block starts with a line for the title ('# Title :') and ends with '# END'.
    - Between these lines, each line should follow the format: '# [field_name] : [field_value]'.
    - After '# END', the data of the experiment is present with the first line as the header.
    - References start with '# Reference :' and end with '# END'.
    - The file should end with a line containing '# END OF FILE'.
    
    If a field is missing, the corresponding attribute in the Experiment object is set to an empty string ('').

    Parameters:
    ------------
    filename : str
        The name of the text file containing the experiments' details.

    Returns:
    ---------
    experiments : list
        A list of Experiment objects populated with the details read from the text file.

    Raises:
    -------
    ValueError:
        If the data in the file is not structured as expected.

    Example:
    --------
    experiments = read_experiments_from_txt('experiment_data.txt')

    Note:
    -----
    The function expects that each Experiment object has attributes corresponding to the fields in the text file.  
    """

    # Create a list of experiments
    experiments = []
    # Open the file
    f = open(filename, 'r')
    # Read the first line 
    line = f.readline()
    # While you dont read '# END OF FILE' read lines
    while True:
        # If the line starts with '# END OF FILE', then you have reached the end of the file
        if line.startswith('# END OF FILE'):
            # Close the file
            f.close()
            # Return the list of experiments
            return experiments
        # While you dont read '# END' read lines
        while not line.startswith('# END'):
            # Create a new experiment
            experiment = Experiment()
            # Read the title exluding any posible '\n' at the end
            experiment.title = line.split(':')[1][:-1]
            # Read the rest of the information. If the information is empty, then set it to an empty string. Exclude any posible '\n' at the end
            line = f.readline()
            experiment.reaction = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            try:
                experiment.Ratio_isomer = float(line.split(':')[1]) 
            except ValueError:
                experiment.Ratio_isomer = ''       
            line = f.readline()
            experiment.quantity = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            experiment.frame = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            try:
                experiment.MF = float(line.split(':')[1])
            except ValueError:
                experiment.MF = ''
            line = f.readline()
            try:
                experiment.MT = float(line.split(':')[1])
            except ValueError:
                experiment.MT = ''
            line = f.readline()
            experiment.X4_ID = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            experiment.X4_code = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            experiment.author = line.split(':')[1][:-1] if line.split(':')[1] != '' else ''
            line = f.readline()
            try:
                experiment.year = int(line.split(':')[1])
            except ValueError:
                experiment.year = ''
            line = f.readline()
            try:
                experiment.data_points = int(line.split(':')[1])
            except ValueError:
                experiment.data_points = ''
            # Read the data. All data starts without '#'. The first line is the header.
            # Read the header
            line = f.readline()
            header = line.split()
            # Read the data
            data = []
            line = f.readline()
            while not line.startswith('#'):
                data.append(line.split())
                line = f.readline()             # In the last iteration, the line is '# Reference   :\n'
            # Create a dataframe with the data
            experiment.data = pd.DataFrame(data, columns=header)
            # Read the references. References start with '# Reference   :\n' and end with '# END'
            line = f.readline()
            experiment.reference = ''
            while not line.startswith('# END'):
                experiment.reference += line
                line = f.readline()             # In the last iteration, the line is '# END'
            # If the reference is empty, then set it to ''. If not, then remove the last '\n'
            if experiment.reference != '': experiment.reference[:-1]
            # Add the experiment to the list of experiments
            experiments.append(experiment)
            # Read the line '# END OF FILE' or next experiment
            line = f.readline()


def read_proton_experiments_from_exfortables(path):
    """
    Reads all proton experiment files located in a given directory and its subdirectories.
    
    The function recursively traverses through all directories and subdirectories under the specified path.
    It reads files that do not have names ending with 'list' or 'ruth' and returns a list of Experiment objects.
    
    Parameters:
    ------------
    path : str
        The absolute or relative path to the root directory containing proton experiment files.
        
    Returns:
    ---------
    experiments : list
        A list of Experiment objects containing data read from the files under the specified path.

    Raises:
    -------
    FileNotFoundError:
        If the specified path does not exist.
    ValueError:
        If the data in the files is not structured as expected.

    Example:
    --------
    experiments = read_proton_experiments_from_exfortables('./proton_experiment_data/')

    Notes:
    ------
    - The function assumes that each Experiment object has attributes corresponding to the fields in the text files.
    - The function relies on `read_experiment()` for reading individual experiment files.
    """

    # Get name of all directories in the path
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    experiments = []

    # Go through all directories
    for d in dirs:
        # Get name of all subdirectories in the path that dont end with 'list'
        subdirs = [sd for sd in os.listdir(os.path.join(path, d)) if os.path.isdir(os.path.join(path, d, sd)) and not sd.endswith('list')]
        # Go through all subdirectories
        for sd in subdirs:
            # Check if there is any subsubdirectory inside the subdirectory
            contains_subsubdir = False
            for item in os.scandir(os.path.join(path, d, sd)):
                if item.is_dir():
                    contains_subsubdir = True
                    break
            # If there is no subsubdirectory, then the files are in the subdirectory
            if contains_subsubdir:
                # Get name of all subdirectories in the path
                subsubdirs = [ssd for ssd in os.listdir(os.path.join(path, d, sd)) if os.path.isdir(os.path.join(path, d, sd, ssd))]
                # Go through all subdirectories
                for ssd in subsubdirs:
                    # Get the name of all the files which dont end with 'list' or 'ruth'
                    files = [f for f in os.listdir(os.path.join(path, d, sd, ssd)) if os.path.isfile(os.path.join(path, d, sd, ssd, f)) and not f.endswith('list') and not f.endswith('ruth')]
                    # Go through all files
                    for f in files:
                        # Read the experiment
                        experiment = read_experiment(os.path.join(path, d, sd, ssd, f))
                        # Add the experiment to the list of experiments
                        experiments.append(experiment)
            else:
                # Get the name of all the files which dont end with 'list' or 'ruth'
                files = [f for f in os.listdir(os.path.join(path, d, sd)) if os.path.isfile(os.path.join(path, d, sd, f)) and not f.endswith('list') and not f.endswith('ruth')]
                # Go through all files
                for f in files:
                    # Read the experiment
                    experiment = read_experiment(os.path.join(path, d, sd, f))
                    # Add the experiment to the list of experiments
                    experiments.append(experiment)
    return experiments


def filter_experiments(experiments, attribute, value):
    """
    Filters a list of Experiment objects based on the specified attribute and its value.
    
    The function evaluates the attribute for each Experiment object in the list, keeping only 
    those objects where the attribute matches the given value. If the attribute does not exist, 
    it will return None and print a list of available attributes.
    
    Parameters:
    ------------
    experiments : list
        A list of Experiment objects to filter.
    attribute : str
        The attribute name based on which the filtering is to be done.
    value : str | int | float
        The value of the attribute for filtering the list of Experiment objects.

    Returns:
    ---------
    filtered_list : list | None
        A list of Experiment objects that have the specified attribute and value.
        Returns None if the attribute does not exist or no experiments meet the condition.

    Raises:
    -------
    AttributeError:
        If the attribute does not exist in the Experiment objects.
        
    Example:
    --------
    filtered_exp = filter_experiments(experiment_list, 'year', 2021)

    Notes:
    ------
    - Make sure the attribute exists in the Experiment object.
    - This function is case-sensitive for string attributes.
    """

    # Check if the attribute exists
    if not hasattr(experiments[0], attribute):
        print('Attribute {} does not exist\n'.format(attribute))
        # Print the list of available attributes
        print('Available attributes:')
        print([attr for attr in dir(experiments[0]) if not attr.startswith('__')])
        return None
    # Filter the list of experiments to get only the ones with the given attribute and value
    filtered_list = [obj for obj in experiments if getattr(obj, attribute) == value]
    
    if len(filtered_list) == 0:
        print('No experiments with {} = {}\n'.format(attribute, value))
        # Print the list of available values for the given attribute
        print('Available values for {}:'.format(attribute))
        print(list(np.unique([getattr(obj, attribute) for obj in experiments])))
        return None
    
    else:
        print('{} experiments with {} = {}'.format(len(filtered_list), attribute, value))
    
    # Return the filtered list
    return filtered_list


def get_unique_values(experiments, attribute):
    """
    Retrieves the distinct values for a specified attribute from a list of Experiment objects.

    The function examines each Experiment object in the list to identify all the unique values 
    of the specified attribute. If the attribute is not present, it returns None and displays 
    a list of available attributes for the Experiment objects.
    
    Parameters:
    ------------
    experiments : list
        A list of Experiment objects from which to extract unique values of the attribute.
    attribute : str
        The name of the attribute whose unique values are to be determined.

    Returns:
    ---------
    unique_values : list | None
        A list of unique values for the given attribute.
        Returns None if the attribute does not exist within the Experiment objects.

    Raises:
    -------
    AttributeError:
        If the attribute does not exist in the Experiment objects.

    Example:
    --------
    unique_years = get_unique_values(experiment_list, 'year')

    Notes:
    ------
    - Ensure the attribute exists in the Experiment objects for accurate results.
    - The function employs the numpy 'unique' function for identifying unique values.
    """

    # Check if the attribute exists
    if not hasattr(experiments[0], attribute):
        print('Attribute {} does not exist\n'.format(attribute))
        # Print the list of available attributes
        print('Available attributes:')
        print([attr for attr in dir(experiments[0]) if not attr.startswith('__')])
        return None
    # Get the unique values of the attribute
    unique_values = list(np.unique([getattr(obj, attribute) for obj in experiments]))
    # Return the unique values
    return unique_values


def clean_dataframe(df, uncertainties=False):
    """
    Cleans a DataFrame by removing redundant columns for analysis.

    This function eliminates all columns with only one unique value and, 
    if the `uncertainties` flag is False, all columns starting with 'd'.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be cleaned.
    uncertainties : bool, optional
        Whether to retain columns that start with 'd'. Default is False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the redundant columns removed.

    Example:
    --------
    cleaned_df = clean_dataframe(original_df)

    Notes:
    ------
    - The function relies on pandas for DataFrame operations.
    """
    # Drop the columns which heading starts with 'd'
    if not uncertainties: df = df.drop([col for col in df.columns if col.startswith('d')], axis=1)

    # Get the number of unique values per column
    unique_values = df.apply(lambda x: len(x.unique()))
    # Get the columns with only one unique value
    columns_to_drop = unique_values[unique_values == 1].index
    # Drop the columns
    df = df.drop(columns_to_drop, axis=1)
    # Return the cleaned dataframe
    return df


def classify_experiments(experiments, attribute):
    """
    Classifies a list of Experiment objects based on a specified attribute.
    
    The function generates individual DataFrames for each unique value of the given attribute
    and writes these as CSV files. Finally, it returns a dictionary containing all these 
    DataFrames.

    Parameters:
    -----------
    experiments : list
        A list of Experiment objects to be classified.
    attribute : str
        The attribute based on which the classification is performed.

    Returns:
    --------
    dfs : dict of pd.DataFrame
        A dictionary with keys as the unique values of the attribute and the corresponding
        DataFrames as values.

    Example:
    --------
    classified_dfs = classify_experiments(experiment_list, 'year')

    Notes:
    ------
    - The function employs the `get_unique_values` function to identify unique attribute values.
    - The output DataFrames are written as CSV files in a directory corresponding to the attribute name.
    """
    # Get the unique values of the attribute
    unique_values = get_unique_values(experiments, attribute)

    # Create an empty dataframe per unique value of the attribute
    dfs = {}
    for value in unique_values:
        dfs[value] = pd.DataFrame()

    # Go through all experiments
    for experiment in experiments:
        # Get the value of the attribute
        value = getattr(experiment, attribute)
        # Add the experiment to the dataframe with the corresponding value of the attribute
        dfs[value] = pd.concat([dfs[value], experiment.to_dataframe()]).reset_index(drop=True)

    # Create a new csv file for each dataframe
    for value in unique_values:
        dfs[value].to_csv('EXFOR_ProtonReactions_Classified_by_{}/{}_{}.csv'.format(attribute, attribute, value))

    # Return the dataframe
    return dfs


def classify_experiments_by_data(experiments):
    """
    Classifies a list of Experiment objects based on a specified attribute.
    
    The function generates individual DataFrames for each unique value of the given attribute
    and writes these as CSV files. Finally, it returns a dictionary containing all these 
    DataFrames.

    Parameters:
    -----------
    experiments : list
        A list of Experiment objects to be classified.
    attribute : str
        The attribute based on which the classification is performed.

    Returns:
    --------
    dfs : dict of pd.DataFrame
        A dictionary with keys as the unique values of the attribute and the corresponding
        DataFrames as values.

    Example:
    --------
    classified_dfs = classify_experiments(experiment_list, 'year')

    Notes:
    ------
    - The function employs the `get_unique_values` function to identify unique attribute values.
    - The output DataFrames are written as CSV files in a directory corresponding to the attribute name.
    """
    # Dictionary to store dataframes grouped by their column headers
    grouped_dataframes = {}
        
    total_experiments = len(experiments)
    print(f"Processing {total_experiments} experiments...")
        
    for idx, experiment in enumerate(experiments, start=1):
        print(f"Processing experiment {idx} out of {total_experiments}...")
            
        # Prepare the data
        df = experiment.prepare_data()
            
        # Create a key based on the column headers
        key = tuple(df.columns)
            
        # If the key already exists in the dictionary, concatenate the dataframe. Otherwise, create a new entry.
        if key in grouped_dataframes:
            grouped_dataframes[key] = pd.concat([grouped_dataframes[key], df], axis=0)
        else:
            grouped_dataframes[key] = df
        
    # Display the detected groups of headers on the screen
    group_number = 1
    for key in grouped_dataframes.keys():
        print(f"\nGroup {group_number}:")
        print(", ".join(key))
        print("="*50)
        group_number += 1

    # Save each grouped dataframe to a CSV file
    total_groups = len(grouped_dataframes)
    print(f"\nSaving {total_groups} grouped dataframes to CSV files...")
    for idx, (key, df) in enumerate(grouped_dataframes.items(), start=1):
        filename = f"EXFOR_ProtonReactions_Classified_Group_{idx}.csv"
        df.to_csv(filename, index=False)
        print(f"Group {idx}'s dataframe saved as {filename}")

    print(f"\nFinished! {len(grouped_dataframes)} groups of experiments found based on column headers.")

    

def plot_experiments(experiments, xlog=False, ylog=False, fig_size=(9,6)):
    """
    Plots a list of experiments using matplotlib and seaborn libraries.

    The function accepts a list of experiment objects and plots them on a 2D graph based on
    their data attributes. It can plot with or without error bars along the x and y axes.
    Additionally, it supports logarithmic scaling for both axes and allows figure size customization.

    Parameters:
    -----------
    experiments : list
        A list of experiment objects to be plotted. Each object should have a 'data' attribute 
        which is a pandas DataFrame containing the relevant data.
    xlog : bool, optional
        A flag to specify if the x-axis should be logarithmic. Default is False.
    ylog : bool, optional
        A flag to specify if the y-axis should be logarithmic. Default is False.
    fig_size : tuple, optional
        The dimensions of the figure to be plotted. Default is (9, 6).

    Returns:
    --------
    None : 
        The function does not return anything; it generates and displays the plot.

    Example:
    --------
    plot_experiments([experiment1, experiment2], xlog=True, ylog=False, fig_size=(10,8))

    Notes:
    ------
    - The function first validates that all experiments have the same DataFrame columns.
    - The plotting accounts for possible error attributes in the x and y axes.
    - The resulting plot can be customized through optional parameters like `xlog`, `ylog`, and `fig_size`.
    """
    
    # First, check if there are experiments to plot
    if not experiments:
        print('No experiments to plot')
        return
    
    # Then, verify that all dataframes have the same headers
    headers = list(experiments[0].data.columns.values)
    for experiment in experiments[1:]:
        if list(experiment.data.columns.values) != headers:
            print('Mismatch in headers between experiments.')
            return
    
    # Now proceed to set up and plot each experiment
    plt.figure(figsize=fig_size)
    for experiment in experiments:
        data = experiment.data
        y_err = data.iloc[:,2].isnull().values.all() or data.iloc[:,2].eq(0).all()
        x_err = data.iloc[:,3].isnull().values.all() or data.iloc[:,3].eq(0).all()
        
        # Depending on y_err and x_err, we decide how to plot
        if y_err == False and x_err == False:
            plt.errorbar(x=data[headers[0]], y=data[headers[1]], 
                        xerr=data[headers[3]], yerr=data[headers[2]],
                        fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
        elif y_err == False:
            plt.errorbar(x=data[headers[0]], y=data[headers[1]], yerr=data[headers[2]], 
                        fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
        elif x_err == False:
            plt.errorbar(x=data[headers[0]], y=data[headers[1]], xerr=data[headers[3]], 
                        fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
        else:
            sns.scatterplot(x=headers[0], y=headers[1], data=data, label=experiment.X4_ID)
    
    # Set scale, labels, title, and display the plot
    if xlog: plt.xscale('log')
    if ylog: plt.yscale('log')
    
    plt.title("Experiments", fontsize=18, pad=15)
    plt.xlabel(headers[0], fontsize=16, labelpad=10)
    plt.ylabel(headers[1], fontsize=16, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()


def plot_outliers(outliers_df, experiments, xlog=False, ylog=False, fig_size=(9,6)):
    """
    Plots outliers along with their corresponding experiments using matplotlib and seaborn libraries.

    The function accepts a DataFrame containing outlier data and a list of experiment objects.
    It plots these outliers and the normal data on a 2D graph. Both outlier and normal data points
    can be visualized with or without error bars along the x and y axes. Logarithmic scaling for both axes
    and figure size customization are also supported.

    Parameters:
    -----------
    outliers_df : pd.DataFrame
        A DataFrame containing the outlier data points. Should have an 'X4_ID' column which matches the experiment IDs.
    experiments : list
        A list of experiment objects to be plotted. Each object should have a 'data' attribute which is a pandas DataFrame.
    xlog : bool, optional
        A flag to specify if the x-axis should be logarithmic. Default is False.
    ylog : bool, optional
        A flag to specify if the y-axis should be logarithmic. Default is False.
    fig_size : tuple, optional
        The dimensions of the figure to be plotted. Default is (9, 6).

    Returns:
    --------
    None :
        The function does not return anything; it generates and displays the plot.

    Example:
    --------
    plot_outliers(outliers_dataframe, [experiment1, experiment2], xlog=True, ylog=False, fig_size=(10,8))

    Notes:
    ------
    - The function first identifies the DataFrame columns that are specific to outliers, not present in normal data.
    - It then groups outliers based on these specific columns before plotting.
    - Normal data and outliers from the same experiment are plotted together for better visualization.
    """

    # Get the column names from the 'data' attribute of a corresponding experiment
    example_experiment = next(experiment for experiment in experiments if experiment.X4_ID in outliers_df['X4_ID'].tolist())
    data_columns = example_experiment.data.columns.values.tolist()

    # Get columns from outliers_df that are not in 'data' and are also not 'X4_ID'
    groupby_columns = [col for col in outliers_df.columns if col not in data_columns and col != 'X4_ID']

    grouped_outliers = outliers_df.groupby(groupby_columns)

    print('Grouped outliers according to columns: {}'.format(groupby_columns))

    for _, group in grouped_outliers:
        plt.figure(figsize=fig_size)
        x4_ids = group['X4_ID'].tolist()

        # Add a legend entry for the outliers without actually plotting them
        plt.scatter([], [], color='red', s=50, label='Outliers', marker='x')

        for experiment in experiments:
            if experiment.X4_ID in x4_ids:
                data = experiment.data
                headers = list(data.columns.values)

                y_err = not (data.iloc[:,2].isnull().values.all() or data.iloc[:,2].eq(0).all())
                x_err = not (data.iloc[:,3].isnull().values.all() or data.iloc[:,3].eq(0).all())

                # Plot normal data
                if y_err and x_err:
                    plt.errorbar(x=data[headers[0]], y=data[headers[1]], 
                                xerr=data[headers[3]], yerr=data[headers[2]],
                                fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
                elif y_err:
                    plt.errorbar(x=data[headers[0]], y=data[headers[1]], yerr=data[headers[2]], 
                                fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
                elif x_err:
                    plt.errorbar(x=data[headers[0]], y=data[headers[1]], xerr=data[headers[3]], 
                                fmt='o', ecolor='black', capsize=3, elinewidth=1,  markersize=5, label=experiment.X4_ID)
                else:
                    plt.scatter(x=data[headers[0]], y=data[headers[1]], s=25, label=experiment.X4_ID)

        # Actually plot the outliers
        for experiment in experiments:
            if experiment.X4_ID in x4_ids:
                specific_outliers = outliers_df[outliers_df['X4_ID'] == experiment.X4_ID]
                plt.scatter(specific_outliers[headers[0]], specific_outliers[headers[1]], color='red', s=50, zorder=3, marker='x')

        if xlog: plt.xscale('log')
        if ylog: plt.yscale('log')
        plt.title("Outliers in Experiments", fontsize=18, pad=15)
        plt.xlabel(headers[0], fontsize=16, labelpad=10)
        plt.ylabel(headers[1], fontsize=16, labelpad=10)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()
        plt.show()