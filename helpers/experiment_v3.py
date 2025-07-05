"""

Author: Austin J. Schleusner
Date: 2025-July-1

Description: This script was made to assist data acquisition for experiments on the Helios
fridge. This is loosely a development on my "experiment_v2.py" script and Camille Mikolas'
"experiment_acq_CM3.py" script. Both of these scripts in themselves were based on code by
Niyaz Beysengulov. My past script ("experiment_v2.py") was used for the high-frequency
Bragg-Cherenkov experiment but needed updating for collecting data on Camille's channel and
resonator device. My hope is that much of this code can be used again when I start
experiments on my next generation device (the SQUILL device). In general, this script can
run measurments on the Yokogawas and Lock-ins as well as on the VNA. As has beem the case
with the eHe scripts in the past, we use SQLite databases to store data and GPIB
communication to control and read most instruments.  It is worth noting that Copilot was
very helpful in explaining to me different elements of this code and how I could improve
upon the previous scripts. Though I want to be clear that I am not a scrub that blindly
asks AI to write my code for me.  At least not yet.

"""

###########################################################################################
## Imports ----------------------------------------------------------------------------- ##
###########################################################################################

# Here we import the necessary libraries and modules to run these experiments.  If an error
# is thrown here, it likely means the module needs to be added to the kernel.

import numpy as np
import warnings
import os.path
import os
import logging
import pyvisa

from time import sleep, strftime
from tabulate import tabulate
from tqdm import tqdm
from helpers.database2 import Create_DB
from newinstruments.BlueFors import BlueFors
bluefors = BlueFors()
from plot_setup import live_plot as lp
from matplotlib import pyplot as plt
from IPython.display import display, HTML, clear_output


###########################################################################################
## Connections ------------------------------------------------------------------------- ##
###########################################################################################

# This is where the code will attempt to connect to the vector network analyzer (VNA), the
# Pfeiffer DPG202 pressure gauge, the two lock-in amplifiers (SR830, SR844), the two
# Agilent 33500 sources, and the many Yokogawas. If these instruments are not connected,
# the code will continue without them, but the absense will be noted in the table.

# Instrument connection summary
instrument_status = []

# Import the instrument drivers 
from newinstruments.vna_E5071_2 import *
from newinstruments.DPG202 import *
from pymeasure.instruments.srs import sr830, sr844
from pymeasure.instruments.yokogawa import yokogawa7651 as y7651
from pymeasure.instruments.yokogawa import yokogawaGS200 as ygs
from pymeasure.instruments.agilent import Agilent33500

# Instantiate function to create a device object and check connectivity
def instantiate(name, cls, address, test_attr=None, printing=False):
    try:
        # Create the device object with the given class and address
        device = cls(address)
        # Start-up of the Yokogawa GS200 and 7651 power supplies
        if isinstance(device, (ygs.YokogawaGS200, y7651.Yokogawa7651)):
            try:
                # Check if the device is already enabled
                if device.source_enabled:
                    # If the device is already enabled, preserve the voltage
                    if printing:
                        print(f"[{name}] output already enabled — preserving voltage")
                # If the device is not enabled, set the source voltage to 0.0 V
                else:
                    device.source_voltage = 0.0
                    # If the device is a 7651, enable the source this way
                    if isinstance(device, y7651.Yokogawa7651):
                        device.enable_source()
                    # Otherwise, the device is a GS200 and enable the source this way
                    else:
                        device.source_enabled = True
                    if printing:
                        print(f"[{name}] output was off — set to 0.0 V and enabled output")
            except Exception as e:
                if printing:
                    print(f"[{name}] failed to check or enable output: {e}")
        # Start-up of the Agilent 33500 function generators
        if isinstance(device, Agilent33500):
            # Set the output to 'off'
            device.output = 'off'
            if printing:
                print(f"[{name}] output disabled at startup")

        # If a test attribute is provided, check its connectivity
        if test_attr:
            _ = getattr(device, test_attr) 
        # If the device is successfully created and connected, append its status as True
        instrument_status.append([name, True, address])
        return device
    except Exception as e:
        # If the device cannot be created or connected, append its status as False
        instrument_status.append([name, False, address])
        # If printing is enabled, print the error message
        if printing:
            print(f"[{name}] Connection failed: {e}")
        return None

# This function connects to all the instruments used in the experiment. It creates global
# variables for each instrument so that they can be accessed throughout the script and in
# the Jupyter notebook. There is also the option to print the 
def connect_instruments(printing=False):
    # global variables to store the instrument objects. This is necessary to make them
    # accessible throughout the script and in the Jupyter notebook.
    global vna, lockin_LF, lockin_HF, dpg
    global yoko_lch, yoko_rgd, yoko_lgt, yoko_rgt, yoko_lres, yoko_mres
    global gen_sign, gen_fila

    # All of the instruments in this experiment (at the moment)
    vna       = instantiate(        "VNA",          E5071_2,  "GPIB0::2::INSTR",        test_attr="get_id", printing=printing)
    lockin_LF = instantiate(      "SR830",      sr830.SR830, "GPIB0::10::INSTR",        test_attr="status", printing=printing)
    lockin_HF = instantiate(      "SR844",      sr844.SR844, "GPIB0::11::INSTR",     test_attr="frequency", printing=printing)
    dpg       = instantiate(     "DPG202",           DPG202,             "COM4", printing=printing)      # No test_attr needed 
    # Yokogawa GS200 and 7651 power supplies
    yoko_lch  = instantiate( "Yoko (lch)", ygs.YokogawaGS200, "GPIB0::7::INSTR",   test_attr="source_mode", printing=printing)
    yoko_rgd  = instantiate( "Yoko (rgd)", ygs.YokogawaGS200, "GPIB0::1::INSTR",   test_attr="source_mode", printing=printing)
    yoko_lgt  = instantiate( "Yoko (lgt)", ygs.YokogawaGS200,"GPIB0::21::INSTR",   test_attr="source_mode", printing=printing)
    yoko_rgt  = instantiate( "Yoko (rgt)", ygs.YokogawaGS200, "GPIB0::6::INSTR",   test_attr="source_mode", printing=printing)
    yoko_lres = instantiate("Yoko (lres)",y7651.Yokogawa7651,"GPIB0::24::INSTR",test_attr="source_voltage", printing=printing)
    yoko_mres = instantiate("Yoko (mres)",y7651.Yokogawa7651,"GPIB0::25::INSTR",test_attr="source_voltage", printing=printing)
    # Agilent 33500 function generators
    gen_sign  = instantiate("33500B (sign)",    Agilent33500,"GPIB0::19::INSTR",         test_attr="shape", printing=printing)
    gen_fila  = instantiate("33500B (fila)",    Agilent33500,"GPIB0::17::INSTR",         test_attr="shape", printing=printing)

# This function returns the controls for the sweep and step parameters and is used
# extensively in the init_reads function.
def get_controls(source, keys):
    return {key: source.get(key) for key in keys}


###########################################################################################
## Connection Table -------------------------------------------------------------------- ##
###########################################################################################

# This function displays the connection status of the instruments in an HTML table with the
# intention of providing a status update at the end of the ipynb imports.

def show_connection_table(status_list):
    # Create the header row of the HTML table
    html = """
    <table style="font-size: 20px; border-collapse: collapse; font-family: monospace;">
        <tr>
            <th class="header">Instrument</th>
            <th class="header">Connected</th>
            <th class="header">Address</th>
        </tr>

    <style>
        .header {
            border: 1px solid #ccc;
            padding: 8px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """
    # Loop through the status list and create an HTML row for each instrument
    for name, connected, address in status_list:
        # Set the color to green if connected, red if not
        color = "#4CAF50" if connected else "#F44336"
        # Use a check mark for connected and a cross for not connected
        symbol = "✔" if connected else "✘"
        # Set the style for the HTML table cells
        td_style = 'border:1px solid #ccc; padding:6px; text-align:center;'
        # Create the HTML row with the instrument name, symbol, and address
        html += f"""
        <tr>
            <td style="{td_style}">{name}</td>
            <td style="{td_style}; color: {color}; font-weight: bold;">{symbol}</td>
            <td style="{td_style}">{address}</td>
        </tr>
        """
    html += "</table>"
    display(HTML(html))


###########################################################################################
## Data Management --------------------------------------------------------------------- ##
###########################################################################################

# This is where we define the functions that will be used to create the sweep lists and
# store data in the SQLite database.

# This creates the list of values being swept over.  The list can be linear or log.
def create_sweep_list(s1=0, s2=1, num=10, scale='linear'):
    try:
        if scale == 'linear':
            sweep_list = np.linspace(s1, s2, num, endpoint=True)
        elif scale == 'log':
            sweep_list = np.logspace(np.log10(s1), np.log10(s2), num, endpoint=True)
        else:
            raise ValueError('Scale must be linear or log')
        return sweep_list
    except:
        print('Sweep list inputs must be: low bound, high bound, point number, scale')
        return None

# Split the filename and suffix (digit), this is a private variable (__)
def __split_name_suffix(name: str):
    # Start at the last character of the name string
    i = len(name) - 1
    # Loop backwards through the name until we find a character that is not a digit
    while i >= 0 and name[i].isdigit():
        i -= 1
    # Earlier characters are the base name
    base = name[:i+1]
    # Later characters are the suffix
    suffix = name[i+1:]
    # Return the suffix as an integer if it exists, otherwise return 0
    return base, int(suffix) if suffix else 0

# Below creates the path for storing the data and the filename to a SQLite database. This
# will automatically create the filename with the date in a folder with the date.
def create_path_filename(measurement_name: str) -> str:
    # Get the date-stamp for directory organization
    date_str = strftime("%Y-%m-%d")
    # Join this date-stamped directory with the larger 'data' file
    subdir = os.path.join('data', date_str)
    # Make a new directory only if it does not already exist
    os.makedirs(subdir, exist_ok=True)
    # Split the numeric suffix from the measurement name
    base, suf = __split_name_suffix(measurement_name)
    # Loop through file names to find the next available filename
    while True:
        # Name the file date_base-suffix.db unless the suffix is 0
        filename = f"{date_str}_{base}-{suf}.db" if suf > 0 else f"{date_str}_{base}.db"
        # Check if the file already exists in the subdir
        filepath = os.path.join(subdir, filename)
        # If the file does not exist, break the loop
        if not os.path.isfile(filepath):
            break
        # If the file exists, increment the suffix and try again
        suf += 1
    # If the suffix is greater than 0, print a message indicating a filename conflict    
    if suf > 0:
        print(f"Avoiding overwrite, new filename: {filename}")
    # Return the full file path
    return filepath

# Scale units to their more readable form. This is primarily used for the 
# tabulate table that displays the class attributes. It will convert units like Hz to
# MHz, kHz, or GHz depending on the value.
def scale_units_to_readable(value, unit):
    # Define the scaling factors and their corresponding units
    scales = {
        'Hz':   [(1e9,  'GHz'), (1e6, 'MHz'), (1e3, 'kHz')],
        'V':    [(1e-3,  'mV'), (1e0, 'V')],
        'A':    [(1e-3,  'mA'), (1e0, 'A')],
        's':    [(1e-9,  'ns'), (1e-6, 'µs'), (1e-3, 'ms'), (1e0, 's')],
        'K':    [(1e-3,  'mK'), (1e0, 'K')],
        'dBm':  [(1e0,  'dBm')],
        'Vpp':  [(1e-3,'mVpp'), (1e0, 'Vpp')],
        'pts':  [(1e0,  'pts')],
        'avgs': [(1e0, 'avgs')]}
    # If the value is a tuple (like the frequency values)...
    if isinstance(value, tuple):
        try:
            # ... iterate through the scaling factors and their corresponding units.
            for factor, new_unit in scales.get(unit, []):
                # scale the values within the tuple by the factor
                scaled = tuple(v / factor for v in value)
                # If all scaled values are within the range of 1 to 1000...
                if all(1 <= abs(v) < 1000 for v in scaled):
                    # ... return it with the new unit formatted to 3 significant figures
                    return f"({scaled[0]:.3g}, {scaled[1]:.3g}) {new_unit}"
        except Exception:
            return str(value)
        return f"{value} {unit}"
    # If the value is an integer or a float...
    if isinstance(value, (int, float)):
        # ... iterate through the scaling factors and their corresponding units.
        for factor, new_unit in scales.get(unit, []):
            # scale 'value' by the factor
            scaled = value / factor
            # If the scaled value is within the range of 1 to 1000...
            if 1 <= abs(scaled) < 1000:
                # ... return it with the new unit formatted to 3 significant figures
                return f"{scaled:.3g} {new_unit}"
        return f"{value:.3g} {unit}"  # fallback
    # For everything else, just return the value as it was entered
    return value  

# Apply the control to the device whether it is a method or attribute.
def apply_control(device, method_name, value):
    attr_or_method = getattr(device, method_name)
    # If it is calleable, it is a method
    if callable(attr_or_method):
        attr_or_method(value)
    # If it is not callable, it is an attribute
    else:
        setattr(device, method_name, value)


# Insert sweep_data into the database
def write_sweep_to_database(sqldb, sweep_data, step_indices):
    """
    Parameters:
        sqldb          : The active Create_DB instance.
        sweep_data     : List of measurement rows (data only).
        step_indices   : Optional. If provided, must be same length as sweep_data.
                         Maps each row to its step index in a 2D sweep.
    """
    # Enumerate over the sweep_data to get the index and row data
    for sweep_index, row in tqdm(enumerate(sweep_data),total=len(sweep_data),
                                 desc='Saving to Database'):
        # 1D case: set step_index = 0
        if step_indices is None:
            step_index = 0
        else:
            step_index = step_indices[sweep_index]
        # Create a full row with sweep_index, step_index, and the row data
        full_row = (step_index, sweep_index) + tuple(row)
        sqldb.insert_data_byrow('table_data', full_row)


###########################################################################################   
## Experiment Class -------------------------------------------------------------------- ##
###########################################################################################

# The experiment class is used for controlling instruments and running experiments. As a 
# general note, variables defined in the class are prefixed with "__" to make them private,
# and should not be accessed directly. Instead, we use the class methods to access and
# modify them.

class exp3():
    # Comments show up in the experimental parameters table
    comment1 = None
    comment2 = None
    comment3 = None
    tconst   = 0.1


    #######################################################################################
    ## General Setup Definitions ------------------------------------------------------- ##
    #######################################################################################

    # The "initializer method" is used at the start of every class and sets up instance
    # attributes and other necessary parameters.
    def __init__(self, 
                 ctrl_instrument: dict, 
                  vna_instrument: dict, 
                 read_instrument: dict,
             bluefors_instrument: dict):
        # Set the private variables for the control and readout of instruments.
        self.__reads = read_instrument
        self.__ctrls = ctrl_instrument
        # Set the private variables for the temperature readout.
        self.__bluefors = bluefors_instrument
        # Set the private variable for the VNA, this needs to be before being 
        # called in the reset_instruments method.
        self.__vna = vna_instrument
        # Reset the instruments to their default states
        self.reset_instruments()

    # This method is used to set the class attributes. It overrides the default
    def __setattr__(self, name, value):
        # Route to control dictionary if key exists
        if hasattr(self, '_exp3__ctrls') and name in self.__ctrls:
            self.__ctrls[name][0] = value
            self.__apply_param(self.__ctrls, name)
        # Route to VNA dictionary
        elif hasattr(self, '_exp3__vna') and name in self.__vna:
            self.__vna[name][0] = value
            self.__apply_param(self.__vna, name)
        # Route to a standard attribute
        else:
            super().__setattr__(name, value)

    # This method is used to get the class attributes. It overrides the default
    def __apply_param(self, source: dict, key: str):
        entry = source[key]
        value = entry[0]  # The value to set
        device = entry[1]  # The instrument object
        # Adjust method index based on length
        method = entry[3] if len(entry) > 4 else entry[2]
        if hasattr(device, method):
            try:
                args = value if isinstance(value, (tuple, list)) else (value,)
                    # If the method is 'ramp_to_voltage', check if the device has a
                    # source_voltage method.
                if method == 'ramp_to_voltage' and hasattr(device, 'source_voltage'):
                    # If the device has a source_voltage, check if the current voltage
                    # is already close to the target voltage (args[0]). If so, skip ramp.
                    current_voltage = device.source_voltage
                    if abs(current_voltage - args[0]) < 1e-6:
                        return  # skip redundant ramp
                # Otherwise apply the method with the arguments
                getattr(device, method)(*args)
            except Exception as e:
                print(f"[{key}] Error applying: {value} → {method} on {device}: {e}")

    # This method prints the class attributes in a table format using the tabulate library.
    def table(self):
        print(tabulate(self.get_ClassAttributes(), 
                       headers=['Attributes', 'Values'], 
                       tablefmt='simple'))

    # This method resets the instruments to their default states by pulling the
    # attributes.
    def reset_instruments(self):
        # If the vna is attached...
        if self.__vna:
            try: 
                # Pull the VNA instrument attributes to set as keys named "Vkey"
                for Vkey in self.__vna:
                    setattr(self, Vkey, self.__vna[Vkey][0])
            except:
                # {e} is a placeholder for the exception (error) message
                print('Problem initializing VNA keys: {e}')
        # If the vna is not attached, print the below message
        else:
            print('No VNA instrument to initialize')
        # Search for key in the control instruments dict
        for Ckey in self.__ctrls:
            # create class attributes to store all control parameters
            setattr(self, Ckey, self.__ctrls[Ckey][0])
        self.table()  # Print the class attributes in a table format
        
    # This creates a printable table of all the instrument control attributes.
    def get_ClassAttributes(self) -> list:
        # Create empty attribute list to store the attributes
        attr_list = []
        # To avoid duplicates
        seen = set()  
        # Add units from control instrument and vna instrument dictionary
        for source in [self.__ctrls, self.__vna]:
            for key, item in source.items():
                # Avoid duplicates here
                if key not in seen:
                    try:
                        val = item[0]
                        unit = item[-1]
                        display_val = scale_units_to_readable(val, unit)
                        attr_list.append([key, display_val])
                    except Exception as e:
                        attr_list.append([key, f"Error: {e}"])
        # Include the BlueFors temperature attributes
        try:
            for label, (frid, func, unit) in self.__bluefors.items():
                # Get the value of the BlueFors instrument attribute
                val = func(frid)
                display_val = scale_units_to_readable(val, unit)
                attr_list.append([label, display_val])
        except:
            pass
        # Add the comments to the attribute list if they aren't 'None'
        for comment_field in ['comment1', 'comment2', 'comment3']:
            if hasattr(self, comment_field):
                value = getattr(self, comment_field)
                if value not in [None, "None"]:
                    attr_list.append([comment_field, value])
        # Return the list of attributes
        return attr_list

    # Initialization of instruments with the user defined parameters
    def instr_init(self, ramp_time = 2) -> None:
        # initializing the yokogawas
        for key in list(self.__ctrls.keys()):
            attr_value = getattr(self, key)
            link = self.__ctrls[key]
            if type(attr_value) is dict:
                val = attr_value.get('val') - attr_value.get('off')
            else:
                val = attr_value
            set_instrument = getattr(link[1], link[3])
            try:
                set_instrument(val, ramp_time)
            except:
                set_instrument(val)
        # initializing the vna
        try:
            for key in list(self.__vna.keys()):
                attr_value = getattr(self, key)
                link = self.__vna[key]
                set_instrument = getattr(link[1], link[2])
                if not isinstance(attr_value, (list, tuple)):
                    set_instrument(attr_value)
                else:
                    set_instrument(*attr_value)
        except:
            print('no VNA instrument to initialize')
        sleep(ramp_time)
        try:
            vna.auto_scale(channel = 1)
        except:
            pass
        print('instruments are initialized!')


    #######################################################################################
    ## Sweep and Step Definitions ------------------------------------------------------ ##
    #######################################################################################

    # Private variable to return the sweep information as a string
    def __sweep_info(self, sw_type, s1, s2, num, scale, offset):
        return f"{sw_type} / {s1:,} : {s2:,} / num={num} / {scale} / off={offset:,}"
    
    # Private variable to identify the object type and return its value
    def __obj_type(self, obj, index):
        if type(obj) is list:
            obj_value = obj[index]
        else:
            obj_value = obj
        return obj_value

    # Variable to set up the start, end, scale type, and offset for the sweep or step
    def control_variables(self, control_type = 'sweep', var = 'None', s1 = 0, s2 = 1,
                          num = 1, scale = 'linear', offset = 0):
        """
        setting up how the instruments will be controlled throughout an experiment
        loop/step/sweep.
        Parameters: 
        control_type: str
            Either 'sweep' or 'step'
        var: None or single string or list of strings 
            The instrument(s) to control during the sweep or step
        s1: int or float
            Start value of the sweep
        s2: int or float
            End value of the sweep
        num: int
            Number of points in the sweep or step
        scale: str
            Linear or logarithmic scale
        offset: int or float
            Offset value to subtract from the sweep or step values
        """
        # If var is not a list, convert it to a list
        if type(var) is not list:
            var = [var]
        # Make an empty control list
        control_lists = []
        # Check that the number of offsets matches the number of variables
        if var and len(var) != len(offset):
            print('''
                  Problem: The number of offsets does not match the number of variables
                  ''')
            return
        # Append sweep lists to the control list for each variable
        for i in range(len(var)):
            first      = self.__obj_type(s1, i)
            final      = self.__obj_type(s2, i)
            scale_type = self.__obj_type(scale, i)
            off        = self.__obj_type(offset, i)
            sweep_list = create_sweep_list(first, final, num, scale_type) + off
            control_lists.append(sweep_list)
            # Store the sweep information in a separate log
            if not hasattr(self, '_sweep_info_log'):
                self._sweep_info_log = {}
            self._sweep_info_log[var[i]] = self.__sweep_info(control_type, first, final,
                                                              num, scale_type, off)
        # Return the variable names, control lists, and number of points. Since I removed
        # the '__indexing_parameters' function, I need to convert 'num' to a list.
        return var, control_lists, [num]
    
    # Public methods to set sweep parameters
    def sweep_params(self, **kwargs) -> None:
        var, control_lists, num = self.control_variables(control_type='sweep', **kwargs)
        self.__sweep = {
            'variable' : var,
            'sweep lists' : control_lists,
            'num points' : num
        }

    # Public methods to set step parameters
    def step_params(self, **kwargs) -> None:
        if kwargs is None:
           var, control_lists, num = ['None'], [[0]], 1
        else: 
            var,control_lists,num = self.control_variables(control_type='step', **kwargs)
        self.__step = {
            'variable' : var,
            'step lists' : control_lists,
            'num points' : num
        }


    #######################################################################################
    ## SQL Database Definitions -------------------------------------------------------- ##
    #######################################################################################
   
    # This function creates a new SQLite database for the experiment with the given name
    # and data columns.  Of note is that the " -> Create_DB" indicates that this function
    # will return an instance of the Create_DB class, which is defined in the separate
    # database2.py file.

    def create_sqldb(self, exp_name: str, data_columns: list) -> Create_DB:
        """
        Creates a new SQLite database file for storing sweep results.

        Parameters:
            exp_name (str): Name of the experiment, used to generate the file path.
            data_columns (list): List of column names for the measurement data.

        Returns:
            Create_DB: An instance of the database interface ready for data insertion.
        """
        # Format the data set name and location in accordance with the create_path_filename
        # function defined above.
        filepath = create_path_filename(exp_name)
        # Create the SQLite database using the Create_DB class
        return Create_DB(
            filepath,
            self.__sweep,
            self.__step,
            data_columns,
            self.get_ClassAttributes(),
        )
    
    # This function closes the SQLite database connection
    def close_sqldb(self, sqldb: Create_DB) -> None:
        sqldb.sql_close()


    #######################################################################################
    ## VNA ----------------------------------------------------------------------------- ##
    #######################################################################################

    # Function to set the VNA wait time based on averaging state and sweep time
    def vna_wait_time(self, vna=None):
        if vna is None:
            vna = self.__vna['instrument']
        # Check if the vna is averaging (True) or not (False)
        vna_avg = vna.get_average_state()
        # Check how many averages are being taken by the vna
        vna_num_avgs = vna.get_averages()
        # If no averaging is occurring...
        if not vna_avg:
            # ... set the sleep time to the sweep time of the vna
            vsleep  = vna.get_sweep_time(channel = 1)
            avstate = False
        # If averaging is occurring...
        elif vna_avg:
            # ... set the sleep time to the sweep time multiplied by the number of averages
            vsleep  = vna.get_sweep_time(channel = 1) * vna_num_avgs
            avstate = True
        # Automatically scale the VNA y axis 
        vna.auto_scale(channel = 1)
        return avstate, vsleep

    # Function to define the VNA data columns depending on the format
    def vna_data_info(self,vna=None):
        if vna is None:
            vna = self.__vna['instrument']
        form_val = vna.get_format()
        # Format options are:
        # MLOGarithmic | PHASe        | GDELay 
        # SLINear      | SLOGarithmic | SCOMplex
        # SMITh        | SADMittance  | PLINear
        # PLOGarithmic | POLar        | MLINear
        # SWR          |REAL          | IMAGinary
        # UPHase       |PPHase
        # These can be selected with the vna's set_format() method

        # Check if the format is one of the VNA data formats that requires two data columns
        if any(s in form_val for s in ['SMIT', 'POL', 'SADM']):
            vna_dat_items = ['VNA1', 'VNA2']
        else:
            vna_dat_items = ['VNA1']
        return vna_dat_items
    
    # Prepare the VNA for measurement by setting the averaging state and waiting for the
    # sweep to complete. 
    def vna_meas_wait(self,vna=None):
        if vna is None:
            vna = self.__vna['instrument']
        avs, vsleep = self.vna_wait_time(vna)
        # Toggle the average state of the VNA to ensure it is ready for the next
        # sweep.
        vna.set_average_state(not avs)
        vna.set_average_state(avs)
        # Let the VNA collect data for the length of time necessary for a full sweep
        # (with possible averaging). The factor of 1.2 is just a safety factor to ensure
        # the VNA has enough time to collect the data we specifically want. 
        sleep(1.2*vsleep)

    # This function pulls the VNA data from the read_dict and returns it as a list.
    def pull_vna_data(self, read_dict):
        # Create an empty list to store the VNA data
        vna_arr = []
        for key, value in read_dict.items():
            device, method, unit = value
            # Get the method or property from the device
            attr = getattr(device, method)
            # Call it if it's a method
            data = attr() if callable(attr) else attr
            vna_arr.append(data)
        return vna_arr


    #######################################################################################
    ## Transport ----------------------------------------------------------------------- ##
    #######################################################################################

    # This function pulls data from the sr830 (LF) or sr844 (HF) and returns it as a list.
    def pull_lockin_data(self, read_dict, target_labels):
        # Create an empty list to store the lockin data
        lockin_arr = []
        # Loop through all read_dict items
        for label in target_labels:
            instr, method, units = read_dict[label]
            # Call the method on the instrument to get the data
            value = getattr(instr, method)
            # If the value is callable, call it to get the data
            data = value() if callable(value) else value
            # Append the data to the lockin_arr list
            if isinstance(data, (tuple, list)):
                lockin_arr.extend(data)
            else:
                lockin_arr.append(data)
        # Return both the x and y data from the selected lock-in amplifier
        return lockin_arr


    #######################################################################################
    ## Querying Control Instruments ---------------------------------------------------- ##
    #######################################################################################

    # This function checks if the instrument is a transport instrument or a vna instrument.
    def query_control_instr(self, instr):
        # Return 'transport' if the instrument is in the transport control instruments
        # dictionary.
        if instr in self.__ctrls:
            return 'transport'
        # Return 'vna' if the instrument is in the vna instruments dictionary
        elif instr in self.__vna:
            return 'vna'
        # If the instrument is in the reads dictionary, check its type
        elif instr in self.__reads:
            # Create a string on the instrument's address
            inst_repr = repr(self.__reads[instr][0])
            # Check if the string representation contains 'vna'
            if "vna" in inst_repr:
                return 'vna'
            else:
                return 'transport'
        # If not in any dictionary, return: 
        else:
            return 'Instrument not in dictionaries'


    #######################################################################################
    ## Initialize For an Experiment Run ------------------------------------------------ ##
    #######################################################################################

    def init_reads(self, printing=False):
        # Since the instrument 'vna' is called explcitly in this function, this check was
        # added to ensure that the vna instrument is named exactly 'vna' before being
        # called to in this function.  If it is not defined, an error is raised.
        try:
            vna
        except NameError:
            raise RuntimeError('''
                        Instrument 'vna' must be defined before calling init_reads()
                        ''')  


        ###################################################################################
        ## Load Instruments ------------------------------------------------------------ ##
        ###################################################################################

        # Load in the sweep instrument from the sweep parameters
        try:
            instrument_sw = list(self.__sweep['variable'])
        except AttributeError:
            print('Problem: \'sweep_params\' has not been defined')
            return # End the function here if AttributeError occurs
        # Check whether the sweep instument is a 'transport' or 'vna' instrument
        sweep_type = self.query_control_instr(instrument_sw[0])
        
        # Load in the step instrument from the step parameters
        try:
            instrument_st = list(self.__step['variable'])
        except AttributeError:
            print('Problem: \'step_params\' has not been defined')
            return # End the function here if AttributeError occurs
        # Check whether the step instrument is a 'transport' or 'vna' instrument
        try:
            step_type = self.query_control_instr(instrument_st[0])
        except:  # There may not be a step instrument used
            step_type = None


        ###################################################################################
        ## Sweep and Step Combinations ------------------------------------------------- ##
        ###################################################################################

        # 1D transport sweep ------------------------------------------------------------ #
        if sweep_type == 'transport' and step_type is None:
            # Create a dictionary of only the read instruments that do not include 'vna.'
            read_keys = [key for key in self.__reads if 'vna' not in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = list(read_dict.keys())
        # 1D vna sweep ------------------------------------------------------------------ #
        elif sweep_type == 'vna' and step_type is None:
            # Create a dictionary of only the read instruments that include 'vna'
            read_keys = [key for key in self.__reads if 'vna' in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = self.vna_data_info(vna)
        # 2D transport sweep ------------------------------------------------------------ #
        elif sweep_type == 'transport' and step_type == 'transport':
            # Create a dictionary of only the read instruments that do not
            # include 'vna.'
            read_keys = [key for key in self.__reads if 'vna' not in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = list(read_dict.keys())
        # 2D hybrid sweep --------------------------------------------------------------- #
        elif sweep_type == 'transport' and step_type == 'vna':
            # Create a dictionary of all read instruments
            read_dict = {key: self.__reads.get(key) for key in self.__reads}
            read_list = list(read_dict.keys()) + self.vna_data_info(vna)
        elif sweep_type == 'vna' and step_type == 'transport':
            # Create a dictionary of all read instruments
            read_dict = {key: self.__reads.get(key) for key in self.__reads}
            read_list = list(read_dict.keys()) + self.vna_data_info(vna)
        # 2D vna sweep ------------------------------------------------------------------ #
        elif sweep_type == 'vna' and step_type == 'vna':
            # Create a dictionary of only the read instruments that include 'vna'
            read_keys = [key for key in self.__reads if 'vna' in key]
            read_dict = {key: self.__reads.get(key) for key in read_keys}
            read_list = self.vna_data_info()
        # Error condition --------------------------------------------------------------- #
        else:
            print('''
            Problem: sweep_type and step_type must be \'transport\', \'vna\' or none
            ''')
            return


        ###################################################################################
        ## Sweep and Step Variables ---------------------------------------------------- ##
        ###################################################################################

        # VNA sweep variable ------------------------------------------------------------ #
        if sweep_type == 'vna':
            # If the sweep variable 'freq_range' appears...
            if 'freq_range' in self.__sweep['variable']:
                # startf is the first entry in the sweep list
                startf  = self.__sweep['sweep lists'][0][0]
                # stopf is the last entry (index = [-1]) in the sweep list
                stopf   = self.__sweep['sweep lists'][0][-1]
                # num_pts is the number of points in the sweep
                num_pts = self.__sweep['num points']
                # Setup frequency range and sweep points for entry to the vna
                self.__vna['freq_range'] = [(startf, stopf), vna, 'set_frequency_range']
                self.__vna['sweep_pts'] = [(num_pts), vna, 'set_sweep_points']
                # sweep_controls is a dictionary of the sweep control inputs
                sweep_controls = get_controls(self.__vna,['freq_range','sweep_pts'])
            else:
                # Otherwise, set the sweep variable as the sweep_control
                sweep_controls = get_controls(self.__vna, self.__sweep['variable'])
            # Get the sweep lists and number of points from the sweep parameters
            sweep_lists = self.__sweep.get('sweep lists')
            num_sweep_points = self.__sweep.get('num points')

        # VNA step variable ------------------------------------------------------------- #
        if step_type == 'vna':
            # If the step variable 'freq_range' appears...
            if 'freq_range' in self.__step['variable']:
                # startf is the first entry in the step list
                startf  = self.__step['sweep lists'][0][0]
                # stopf is the last entry (index = [-1]) in the step list
                stopf   = self.__step['sweep lists'][0][-1]
                # num_pts is the number of points in the step
                num_pts = self.__step['num points']
                # Setup frequency range and step points for entry to the vna
                self.__vna['freq_range'] = [(startf, stopf), vna, 'set_frequency_range']
                # The vna driver uses 'sweep' as opposed to step, so it is kept as
                # 'sweep' here.
                self.__vna['sweep_pts'] = [(num_pts), vna, 'set_sweep_points']
                # step_controls is a dictionary of the step control inputs
                step_controls = get_controls(self.__vna, ['freq_range','sweep_pts'])
            else:
                # Otherwise, set the step variable as the step_control
                step_controls = get_controls(self.__vna, self.__step['variable'])
            # Get the step lists and number of points from the step parameters
            step_lists = self.__step.get('step lists')
            num_step_points = self.__step.get('num points')

        # Transport sweep variable ------------------------------------------------------ #
        if sweep_type == 'transport':
            # Get the sweep controls from the sweep parameters
            sweep_controls = get_controls(self.__ctrls, self.__sweep['variable'])
            # Get the sweep lists and number of points from the sweep parameters
            sweep_lists = self.__sweep.get('sweep lists')
            num_sweep_points = self.__sweep.get('num points')

        # Transport step variable ------------------------------------------------------- #
        if step_type == 'transport':
            # Get the step controls from the step parameters
            step_controls = get_controls(self.__ctrls, self.__step['variable'])
            # Get the step lists and number of points from the step parameters
            step_lists = self.__step.get('step lists')
            num_step_points = self.__step.get('num points')

        # None step variable ------------------------------------------------------------ #
        if step_type is None:
            # If there is no step variable, set the step controls to an empty dictionary
            step_controls = {}
            # Set the step lists and number of points to None
            step_lists = None
            num_step_points = None
       
        # Print variables for diagnositics ---------------------------------------------- #
        if printing:
            print('Read dictionary:', read_dict)
            print('Read list:', read_list)
            print('Sweep type:', sweep_type)
            print('Step type:', step_type)
            print('Sweep controls:', sweep_controls)
            print('Step controls:', step_controls)
            print('Sweep lists:', sweep_lists)
            print('Step lists:', step_lists)

        # Return the variables for run experiment --------------------------------------- #
        return (
            read_dict, 
            read_list,
            sweep_type,
            step_type, 
            sweep_controls, 
            step_controls, 
            sweep_lists, 
            step_lists, 
            num_sweep_points, 
            num_step_points
            )
    

    #######################################################################################         
    ## Generic VNA Sweep --------------------------------------------------------------- ##
    #######################################################################################
    
    def run_vna_sweep(
        self,
        vna,                            # the earlier defined vna instrument
        num_sweep_points,               # number of VNA frequency points
        sweep_var: str,                 # 'power', 'elec_delay', or 'none'
        sweep_list: list = None,        # list of values to sweep over (if applicable)
        sweep_controls: dict = None,    # e.g. {'power': (val0, instr, task)}
        read_dict: dict = None,         # used in pull_vna_data
        step_val=0):                    # optional step tag for outer sweeps
        # The data from the vna sweep will be stored in this list
        sweep_data = []
        # Unpack the number of frequency points 
        num_pts = num_sweep_points[0]
        # Handle the 'freq_range' keyword as a special no-sweep case
        if sweep_var == 'freq_range':
            # S1 will be the start frequency
            start_freq = sweep_list[0]
            # S2 will be the stop frequency
            stop_freq = sweep_list[-1]
            # Apply frequency configuration
            vna.set_frequency_range(start_freq, stop_freq)
            # Apply the number of sweep points
            vna.set_sweep_points(num_pts)
            # Wait for the VNA to finish the sweep
            self.vna_meas_wait(vna)
            # Only pull from the vna read_dict entires
            vna_only_dict = {k: v for k, v in read_dict.items() if k.startswith('vna_')}
            # Pull the VNA data from the read_dict
            vna_arr = self.pull_vna_data(vna_only_dict)
            # Switch to a numpy array for later slicing
            vna_arr = np.array(vna_arr)
            vna_arr = vna_arr.T # Transpose to match frequency points
            # Create a frequency index based on the number of sweep points
            freq_index = np.linspace(start_freq, stop_freq, num_pts, endpoint=True)
            # If step_val is a list or tuple, create a step index for each value
            if isinstance(step_val, (list, tuple)):
                # Create multiple step index columns
                step_index = np.stack([np.full_like(freq_index, val) for val in step_val], axis=1)
            else:
                # Single step variable
                step_index = np.full_like(freq_index, step_val)
            step_index = step_index[:, None]  # Ensure step_index is 2D
            stacked = np.hstack((step_index, freq_index[:, None], vna_arr))
            return stacked.tolist()
        # Sweeping one parameter (power, delay, etc.)
        val0, instr, task, units = sweep_controls[sweep_var]
        for val in tqdm(sweep_list, desc=f'Sweeping {sweep_var}', leave=False):
            apply_control(instr, task, val)
            self.vna_meas_wait(vna)
            vna_arr = self.pull_vna_data(read_dict)
            vna_arr = np.array(vna_arr).T  # Transpose to match frequency points
            freq_pts = vna.get_fpoints(vna)
            freq_index = freq_pts[:,None] # Ensure freq_index is 2D
            sweep_index = np.full((len(freq_pts), 1), val) 
            stacked = np.hstack((freq_index, sweep_index, vna_arr))
            sweep_data.extend(stacked.tolist())
        return sweep_data


    #######################################################################################         
    ## Generic Transport Sweep --------------------------------------------------------- ##
    #######################################################################################

    def run_transport_sweep(
        self, 
        volt_lists, 
        sweep_controls, 
        select_read,
        sweep_order,
        step_val=0
        ):
        # All of the data from the sweep will be stored in this list
        sweep_data = []
        # Check if the 'volt_list' dictionary contains the required variables
        if set(volt_lists) != set(sweep_order):
            print(f"Problem: 'volt_lists' must contain the keys: {set(sweep_order)}")
            return
        # Check if the 'sweep_controls' dictionary contains the required variables
        if set(sweep_controls) != set(sweep_order):
            print(f"Problem: 'sweep_controls' must only contain the keys: {set(sweep_order)}")
            return   
        # Get the sweep voltage lists in the declared 'sweep_order' order
        sweep_lists = [volt_lists[var] for var in sweep_order]
        # Check if all voltage sweep lists are of equal length
        if not all(len(lst) == len(sweep_lists[0]) for lst in sweep_lists):
            print("Problem: All voltage sweep lists must be of equal length")
            return
        # Ramp each variable to its initial value
        for var in sweep_order:
            val0, instr, task, ramp_func, units = sweep_controls[var]
            getattr(instr, ramp_func)(volt_lists[var][0])    
        # Sleep for more than 2 second to allow the above ramping complete
        time.sleep(3)
        # Loop through the paired voltage values in the two lists
        for volts in tqdm(zip(*sweep_lists), desc=f'Sweeping {sweep_order}', leave=False):
            # Apply sweep controls to all sweep variables
            for var, val in zip(sweep_order, volts):
                val0, instr, task, ramp_func, units = sweep_controls[var]
                # Call the method or attribute on the instrument
                apply_control(instr, task, val)
            # Sleep to allow the instruments/setup to settle
            sleep(self.tconst)
            # Pull the lock-in (LF) data from the select_read
            lf_values = self.pull_lockin_data(select_read, target_labels=['Vlfx', 'Vlfy'])
            # Normalize step_value: tuple → unpacked list; scalar → wrap as list
            step_tag = list(step_val) if isinstance(step_val, tuple) else [step_val]
            # Append the voltage and lock-in values to the sweep_data list
            sweep_data.append(step_tag + list(volts) + lf_values)
        return sweep_data


    #######################################################################################
    ## Generalized 2D Helper Function -------------------------------------------------- ##
    #######################################################################################

    def run_2d_helper(
        self,     
        step_map: dict,
        step_controls: dict,
        sweep_func: callable,   # Function being swept (e.g. self.run_transport_sweep_vlch)
        sweep_args: list,       # Arguments for the sweep function
        *,
        step_order: list,       # Order-dependent list of step variable name
        sweep_order: list       # Order-dependent list of sweep variable name
        ):     
        # Create an empty list for the sweep data
        sweep_data = []
        # Create an empty list for step indices
        step_indices = []
        # Get the step list(s) for the step variable(s): This gives a list of tuples
        step_lists = [step_map[var] for var in step_order]
        # zip(*step_lists) combines the lists into tuples, where each tuple contains
        # one value from each list.  Use enumerate to index the step_lists.
        for step_idx, step_vals in tqdm(enumerate(zip(*step_lists)),
                                        total=len(step_lists[0]),
                                        desc=f'Stepping {step_order}',
                                        leave=True):
            # Step each variable in step_vars to the corresponding value in step_vals
            for var, val in zip(step_order, step_vals):
                # Unpack the step_controls for the variable
                val0, instr, task, ramp_func, units = step_controls[var]
                # Set the instrument to the value
                apply_control(instr, task, val)
            # Having set all of the instruments to their values for this step,...
            # Sleep to allow the instruments/setup to settle
            sleep(self.tconst)
            # Run the function of the 'sweep_func' argument with the sweep_args
            out_data = sweep_func(*sweep_args, 
                       step_val=step_vals[0] if len(step_vals) == 1 else step_vals)
            # Extend the sweep_data with the out_data
            sweep_data.extend(out_data)
            # Track step index for each row
            step_indices.extend([step_idx] * len(out_data))
        # Return the sweep data
        return sweep_data, step_indices


    #######################################################################################
    ## Run Experiment ------------------------------------------------------------------ ##
    #######################################################################################

    def run_experiment(self, exp_name = 'sweep_NA', savedata = False, **kwargs):

        # Import definitions from the 'init_reads'
        (
        read_dict, 
        read_list,
        sweep_type,
        step_type, 
        sweep_controls, 
        step_controls, 
        sweep_lists, 
        step_lists, 
        num_sweep_points, 
        num_step_points
        ) = self.init_reads()

        # Begin with the sweep_data set as None
        sweep_data = None
        # Begin with the step_indices set as None
        step_indices = None
        # Create the ordered list of sweep variables 
        sweep_order = list(self.__sweep['variable'])
        # Create the ordered list of step variables 
        step_order = list(self.__step['variable'])
        # Specify the lock-in read dictionaries being used
        if sweep_type == 'transport':
            select_read = {key: read_dict[key] for key in ['Vlfx', 'Vlfy']}

        # Assign the sweep lists to their variables and then package as a dictionary
        sweep_map = dict(zip(self.__sweep['variable'], self.__sweep['sweep lists']))
        # Assign the step lists to their variables and then package as a dictionary
        step_map = dict(zip(self.__step['variable'], self.__step['step lists']))


        ###################################################################################
        ## 1D Sweep Options ------------------------------------------------------------ ##
        ###################################################################################

        if step_type is None:

            # 1D VNA Sweep -------------------------------------------------------------- #
            if sweep_type == 'vna': 
                sweep_data = self.run_vna_sweep(
                    vna=vna,
                    num_sweep_points=num_sweep_points,
                    sweep_var=sweep_order[0],
                    sweep_list=sweep_map.get(sweep_order[0]),
                    sweep_controls=sweep_controls,
                    read_dict=read_dict)
                            
            # 1D Transport Sweep -------------------------------------------------------- #
            elif sweep_type == 'transport':
                sweep_data = self.run_transport_sweep(
                    volt_lists=sweep_map,
                    sweep_controls=sweep_controls,
                    select_read=select_read,
                    sweep_order=sweep_order)
            

        ###################################################################################
        ## VNA Step Options ------------------------------------------------------------ ##
        ###################################################################################

        elif step_type == 'vna':

            # 2D VNA Sweep -------------------------------------------------------------- #

            # Since the 'power' and 'elec_delay' variables are already 2D in the sense that
            # they are also observing a range of frequencies, the only combination that
            # couldbe used for a 2D sweep here is a sweep of power and elec_delay. But this
            # is a combination of things being swept that I will never use.  So there is no
            # need to write any functions for the sweep:VNA, step:VNA combination.

            # 2D Transport Sweep-VNA Step ----------------------------------------------- #

            # As is the general case, whatever instrument is being used as the 'sweep'
            # variable will determine what instrument is being used for data collection. If
            # the sweeping instrument is of the 'transport' type, then the data collection
            # will be done via the lock-in amplifier and if the sweeping instrument is of
            # the 'vna' type, then the data will be collected by the VNA.  This is important
            # here because there is not a use case for using the VNA as an input variable
            # without also wanting it to also collect data.  This is what would happen if 
            # the VNA was used as a step variable and one of the transport instruments was
            # used as the sweep variable.  Thus, this step: vna and sweep: transport
            # combination is not implemented in the code.

            print("""
                  Problem: VNA as a step variable is not presently supported.  The VNA
                  is useful for collecting data but not for stepping through.  As a general
                  rule, the data collection method corresponds to the sweep variable.  For
                  now, there are no known use cases for stepping the VNA and then observing
                  via the transport setup (i.e. the lock-in amplifier).  Similarly, the
                  combination of VNA as both the step and sweep type has not been
                  implemented since there is not a known use case.
                  """)
            return


        ###################################################################################
        ## Transport Step Options ------------------------------------------------------ ##
        ###################################################################################

        elif step_type == 'transport':

            # 2D Transport-VNA Sweep ---------------------------------------------------- #
            if sweep_type == 'vna':
                sweep_var=sweep_order[0]
                sweep_list=sweep_map.get(sweep_order[0])

                sweep_data, step_indices = self.run_2d_helper(
                    step_map=step_map,
                    step_controls=step_controls,
                    sweep_func=self.run_vna_sweep,
                    sweep_args=[vna, num_sweep_points, sweep_var,
                        sweep_list, sweep_controls, read_dict],
                    step_order=step_order,
                    sweep_order=sweep_order)
            
            # 2D Transport-Transport Sweep ---------------------------------------------- #
            elif sweep_type == 'transport':  
                sweep_var=sweep_order
                # Us the run_2D_helper on the generic run_transport_sweep function
                sweep_data, step_indices = self.run_2d_helper(
                    step_map=step_map,
                    step_controls=step_controls,
                    sweep_func=self.run_transport_sweep,
                    sweep_args=[sweep_map, sweep_controls, select_read, sweep_order],
                    step_order=step_order,
                    sweep_order=sweep_order)
 

        # If Unimplemented Step-Sweep is Called ----------------------------------------- #
        else:
            print(f"No sweep logic implemented for step: {step_type}, sweep: {sweep_type}")
            return


        ###################################################################################
        ## Saving Data ----------------------------------------------------------------- ##
        ###################################################################################
        
        # If no sweep_data was collected, print a message and return
        if sweep_data is None:
            print('Problem: No sweep data collected. Check the sweep and step parameters.')
            return

        # Create Column Headers
        if sweep_type == 'vna':
            read_cols = [k for k in read_dict if k.startswith('vna_')]
        elif sweep_type == 'transport':
            read_cols = list(select_read.keys())
        else:
            print('Problem: sweep_type must be \'transport\' or \'vna\'')
            return

        # If sweeping power or elec_delay on the VNA, override the sweep_order
        if sweep_type == 'vna' and sweep_order[0] in ['power', 'elec_delay']:
            # prepend the 'freq_range' to the sweep_order
            sweep_order = ['freq_range'] + sweep_order
        elif sweep_order[0] == 'freq_range' and step_order == []:
            step_order = ['empty step']

        # Construct the full column list
        columns = step_order + sweep_order + read_cols

        # Check that the sweep_data is a list of lists and that each row has the same
        # length as columns.

        for i, row in enumerate(sweep_data):
            if len(row) != len(columns):
                print(f"Row {i}: expected {len(columns)} cols, got {len(row)} → {row}")
        print(f"Non-index columns: {columns}")

        assert all(
            len(row) == len(columns)
            for row in sweep_data
        ), "Problem: Mismatch in data and column lengths"

        # If savedata is True, create a SQLite database and insert the data
        if savedata:
            # Create the database 
            sqldb = self.create_sqldb(exp_name, columns)
            # Save the sweep_data to the database 
            write_sweep_to_database(sqldb, sweep_data, step_indices)
            # End the connection to the database
            self.close_sqldb(sqldb)
            print('Data saved to database')
        else:
            print('Data collected but not saved to database')
        return