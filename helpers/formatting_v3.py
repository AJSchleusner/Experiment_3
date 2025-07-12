'''

Author: Austin J. Schleusner
Date: 2025-July-5

This is code for accessing the database files and simplifying plotting
of data collected by experiment_v3.py and acquisition_v3.ipynb.  Like
that of the experiment code, this is a development upon previous code 
largely written by Niyaz Beysengulov and Camille Mikolas.  My hope is 
to update our plotting code and merge in the parts of database2.py that
we have relied upon to pull data from the database files.  In the 
creation of plots, I plan to incorporate the table structure I used for 
the meta data of my NV-center experiments.  I think it looks clean and 
if you are reading this and disagree, I am sorry that you have poor 
taste.  Like in the experiment code, Microsoft Copilot is helping me 
with the coding, but not to an extent where I am obsolete.

'''



###########################################################################################
## Imports ----------------------------------------------------------------------------- ##
###########################################################################################

import numpy as np
import sqlite3
import pandas as pd
import os
import re
import math

from datetime import date
from newinstruments.BlueFors import BlueFors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.table import Table
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.cm import ScalarMappable
from functools import partial
from scipy.optimize import curve_fit
from pathlib import Path
import textwrap


###########################################################################################
## Accessing the Database Files -------------------------------------------------------- ##
###########################################################################################

class access_db():
    # Initialization method for the access_db class.  This runs when an instance of the
    # class is created.
    def __init__(self, db_path: str):
        # Connect to the SQLite database file
        self.conn = sqlite3.connect(db_path)
    # Collect a list of all tables in the database
    def list_tables(self):
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        return [row[0] for row in self.conn.execute(query).fetchall()]
    # Get the names of all columns in a specific table
    def get_columns(self, table: str):
        cursor = self.conn.execute(f"SELECT * FROM {table} LIMIT 1")
        return [col[0] for col in cursor.description]
    # Get the data from a specific table as a pandas DataFrame
    def grab_table(self, table: str) -> pd.DataFrame:
        return pd.read_sql_query(f"SELECT * FROM {table}", self.conn)
    # Close the connection to the database
    def close_db(self):
        self.conn.close()

# Check if a value is a valid S-parameter string (e.g., "S21", "S11")
def is_s_parameter(value):
    return isinstance(value, str) and re.fullmatch(r"S\d{2}", value.upper()) 

# Extracts a tuple of two floats from a string like '(4.4e9, 4.6e9) set_frequency_range'
def extract_freq_tuple(s):
    match = re.match(r"\(([^,]+),\s*([^)]+)\)", s)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

# Get data from the SQLite database file and return it in a structured format
def load_experiment_data(db_path: str):
    # Automatically append '.db' if it's missing
    if not db_path.endswith(".db"):
        db_path += ".db"
    # Create an instance of the access_db class to connect to the database
    db = access_db(db_path)
    # Check what tables are available in the database
    available_tables = db.list_tables()
    # Define a function to grab data from a specific table
    def grab(name):
        return db.grab_table(name) if name in available_tables else None
    # Access the info table to get metadata about the experiment
    info = grab('table_info')

    # Initialize an empty dictionary to hold metadata
    metadata = {}
    # Extract the headers to set as metadata_col
    for col in info.columns:
        # Get the raw values from the column, dropping any NaN values
        raw_vals = info[col].dropna().tolist()
        # Handle frequency columns: convert to float, convert Hz → GHz
        if "freq_range" in col.lower():
            parsed_vals = [extract_freq_tuple(v) for v in raw_vals]
            parsed_vals = [t for t in parsed_vals if t]  # remove failed parses
            # Flatten each (start, stop) pair into two lines in order
            formatted = []
            for pair in parsed_vals:
                formatted.extend([f"{round(val / 1e9, 3)} GHz" for val in pair])
            metadata[col] = formatted
        # If the column is 'set_vna_meas', extract S-parameters
        elif col == "set_vna_meas":
            match = re.search(r"\bS\d{2}\b", str(raw_vals))
            if match:
                metadata.setdefault(col, []).append(match.group())
        else:
            # Default: treat values as strings
            values = [str(v) for v in raw_vals]
            metadata[col] = values

    # Access the tables containing different types of data
    data = grab('table_data')
    sweep = grab('table_sweep')
    step = grab('table_step')   # Note step will be empty in a 1D measurement
    # Get the headers for the sweep and step tables, if they exist
    sweep_headers = sweep.columns.tolist() if sweep is not None else []
    step_headers  = step.columns.tolist()  if step is not None else []
    # End the connection to the database
    db.close_db()
    # Return a dictionary containing the data, sweep, step, and metadata
    return {
        "data": data.to_numpy() if data is not None else None,
        "sweep": sweep.to_numpy() if sweep is not None else None,
        "step": step.to_numpy() if step is not None else None,
        "metadata": metadata,
        "sweep_headers": sweep_headers,
        "step_headers": step_headers}

# Generate a unique filename (used for saving figures)
def get_unique_filename(save_dir, base_name, extension=".jpg", use_date=True,
                        custom_date=None):
    save_path = Path(save_dir)
    # Get today's date or use a provided one
    date_str = custom_date if custom_date else date.today().isoformat()
    idx = 1
    while True:
        if use_date:
            candidate = f"{date_str}_{base_name}_{idx}{extension}"
        else:
            candidate = f"{base_name}_{idx}{extension}"
        full_path = save_path / candidate
        if not full_path.exists():
            return full_path
        idx += 1

# Try to pull the format from the metadata for later use
def get_vna_format(metadata):
    # Try to call to the format key in the metadata dictionary
    try:
        return metadata['format'][0].upper()
    # If the format key cannot be retrieved
    except KeyError:
        raise KeyError('Missing format key in metadata')


###########################################################################################
## Figure Formation Helpers ------------------------------------------------------------ ##
###########################################################################################

# Format plots for displaying data and the meta data.
def format_plot(fig_wide = 12, fig_tall = 8, left_width = 2.5, right_width = 1):
    fig = plt.figure(figsize= (fig_wide, fig_tall))
    # Create a GridSpec layout with 1 row and 2 columns
    gs = GridSpec(1, 2, width_ratios=[left_width, right_width], figure=fig)
    # Create subplots using the GridSpec layout
    ax_plot = fig.add_subplot(gs[0, 0])  # Left subplot
    ax_meta = fig.add_subplot(gs[0, 1])  # Right subplot
    # Turn off the right subplot's axis
    ax_meta.axis('off')
    return fig, ax_plot, ax_meta

# Comment wrapping definition
def wrap_comment(text, max_chars=60):
    return "\n".join(textwrap.wrap(text, width=max_chars))








# Format plots for displaying data and n metadata tables.
def form_plot(num_meta: int=1, fig_w = 13, fig_h = 8, left_width = 2.5,
              right_width = 1, facecolors=None, titles=None):
    fig = plt.figure(figsize= (fig_w + 2*num_meta - 2, fig_h))
    # Total columns is 1 for data and num_meta for metadata
    total_cols = 1 + num_meta
    right_width = [right_width] * num_meta
    width_ratios = [left_width] + right_width
    # Create a GridSpec layout
    gs = GridSpec(1, total_cols, width_ratios=width_ratios, figure=fig)
    # Create subplots using the GridSpec layout
    ax_plot = fig.add_subplot(gs[0, 0])  # Data subplot
    # Create an empty list to hold metadata axes
    ax_meta_list = []
    # Loop through the number of metadata sections to create subplots
    for i in range(num_meta):
        # Create a new subplot for each metadata section
        ax = fig.add_subplot(gs[0, i + 1])
        ax.axis('off')  # Turn off the axis for metadata subplots
        # Set the face color if provided
        if facecolors and i < len(facecolors):
            ax.set_facecolor(facecolors[i])
        # Title are assigned in the metadata table function
        # Append the axis to the list
        ax_meta_list.append(ax)
    # Return the figure components
    return fig, ax_plot, ax_meta_list


def meta_table(ax, metadata: dict, title: str = 'Metadata', fontsize=12,
             comm_on: bool = False, facecolor='aliceblue', style=None):
    # Make a copy of the metadata to avoid modifying the original
    metadata = metadata.copy()
    # Remove comment fields from metadata
    comment_keys = [k for k in metadata if 'comment' in k.lower() or 'note' in k.lower()]
    comments = [f"{k}: {', '.join(map(str, metadata.pop(k)))}" for k in comment_keys]
    # Clear axes
    ax.clear()
    # Dynamically change box_height based on whether comments are present
    has_comments = comm_on and bool(comments)
    box_height = 0.96 if not has_comments else 0.96 + 0.12  
    # Set the table's title
    ax.set_title(title, fontsize=fontsize+4, loc='center', pad=10)
    ax.axis('off')  # Turn off the axis
    # Compute layout
    key_width = 0.50  # percentage of horizontal space for keys
    value_start = 0.03 + key_width + 0.04  # left padding + key width + spacing
    y_pos = 0.98  # Start near top of box
    line_spacing = 0.98/(len(metadata)+1)  # normalized spacing per line

    # Populate the table with the metadata entires
    for k, v in metadata.items():
        val_lines = v if isinstance(v, list) else [str(v)]
        num_lines = len(val_lines)
        # Key only on first line
        ax.text(0.02, y_pos, k, fontsize=fontsize, fontfamily='monospace',
                ha='left', va='top', transform=ax.transAxes)
        # Each value line on its own row
        for i, line in enumerate(val_lines):
            ax.text(value_start, y_pos - i * line_spacing, line, fontsize=fontsize,
                    fontfamily='monospace', ha='left', va='top', transform=ax.transAxes,
                    clip_on=True)
        # Adjust y position for next key-value pair
        y_pos -= line_spacing * num_lines
        # Border box
        rect = patches.FancyBboxPatch(
            (0, 0.98-box_height), 1, box_height,
            boxstyle="round,pad=0.02",
            linewidth=style['tick_width'] if style else 3,
            edgecolor="black",
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False)
        # Add the rectangle to the table
        ax.add_patch(rect)

    # If comments are present, add a divider line
    if has_comments:
        divider_y = 0
        ax.plot([0.08, 0.92], [divider_y, divider_y],
                transform=ax.transAxes,
                color="black",
                linewidth=style['tick_width'] if style else 2,
                solid_capstyle='round')

    # Render comment separately at the bottom
    if comments and comm_on == True:
        # Merge and wrap all comments into a single string
        comment_text = wrap_comment("\n".join(comments), max_chars=30)
        ax.text(
            0.02,  -0.06,  # Just below the box
            comment_text,
            fontsize=fontsize - 1,
            fontstyle='italic',
            fontfamily="monospace",
            ha='left',
            transform=ax.transAxes,
            clip_on=False,
            linespacing=1.3)


###########################################################################################
## Formatting the Plot ----------------------------------------------------------------- ##
###########################################################################################

# General Format ------------------------------------------------------------------------ #

def general_plt(ax, xlabel=None, ylabel=None, title=None, log_y=None, ylims=None,
                style=None, legend=True):
    # Set the axes labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    # Set the title of the plot
    if title:
        ax.set_title(title, fontsize=style['title_fontsize'], pad=style['title_pad'])
    # Set the tick parameters
    ax.tick_params(axis='both',
                   length=style['tick_length'],
                   width=style['tick_width'],
                   direction='in',
                   labelsize=style['fontsize'],
                   pad=style['tick_pad'])
    # Set the major locators for x and y axes
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Set the spine parameters
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # If true, set the y-axis to logarithmic scale
    if log_y:
        ax.set_yscale('log')
    # If provided, set y limits
    if ylims:
        ax.set_ylim(ylims)
    # If labels are provided, add a legend
    if legend:
        ax.legend(fontsize=style['fontsize'] - 1)


# 1D Plot ------------------------------------------------------------------------------- #
def form_1d_plot(ax, sweep, data, meta, filename=None, sweep_type=None, style=None):
    # Clear the axis before plotting
    ax.clear()

    # Customize plot based on sweep_type
    if sweep_type == 'freq_range':
        # Determine which data type was taken from the VNA
        vna_form = get_vna_format(meta)
        if vna_form in ('MLOG',):
            y_label = "S21 (dB)"
        elif vna_form == 'PHAS':
            y_label = "S21 (degrees)"
        elif vna_form == 'IMAG':
            y_label = "Im(S21)"
        elif vna_form == 'REAL':
            y_label = "Re(S21)"
        else:
            print(f"Unknown VNA format: {vna_form}. Update code to include format.")
            return
        # Scale the VNA sweep from Hz to GHz.  Also for a 'freq_range' sweep, the measured
        # amplitude is stored in data[:,4] (column number 5).
        ax.plot(sweep*1e-9, data[:,4], 
                       color=style['line_color'], 
                       linewidth=style['line_width'])
        # Set freq_range plot settings
        general_plt(ax, xlabel="Frequency (GHz)", ylabel=y_label,
                    title=filename, log_y=False, ylims=None,
                    style=style, legend=False)
    # add other sweep_type conditions as needed









# 2D Plot ------------------------------------------------------------------------------- #
def form_2d_plot(ax, data, meta, filename=None, 
                sweep_type=None, 
                vmin=None, vmax=None, style=None):

    # Clear the axis before plotting
    ax.clear()

    # If the sweep_type is one of the vna options
    if sweep_type in ('freq_range', 'power', 'elec_delay'):
            # Determine which data type was taken from the VNA
        vna_form = get_vna_format(meta)
        if vna_form == 'MLOG':
            z_label = "S21 (dB)"
        elif vna_form == 'PHAS':
            z_label = r"$\phi$ (S21) (degrees)"
        elif vna_form == 'IMAG':
            z_label = "Im(S21) (dB)"
        elif vna_form == 'REAL':
            z_label = "Re(S21) (dB)"
        else:
            print(f"Unknown VNA format: {vna_form}. Update code to include format.")
            return

    # Customize plot based on sweep_type
    if sweep_type == 'power':
        # Select the columns in the data table 
        frequencies  = data[:,5]
        powers       = data[:,3]
        measurements = data[:,4]
        # Find the unique frequencies and powers
        unique_freqs = np.unique(frequencies)
        unique_powers = np.sort(np.unique(powers))
        # Sort the unique powers to ensure the y-axis is in ascending order
        sort_indices = np.argsort(unique_powers)
        # Create a 2D grid for the measurements
        data_z = measurements.reshape(len(unique_powers), len(unique_freqs))
        # If the first power is less than the last, flip the data vertically
        if unique_powers[0] < unique_powers[-1]:
            data_z = data_z[::-1, :] # Flip vertically
        data_x = unique_freqs * 1e-9  # Convert Hz to GHz
        data_y = unique_powers  # Power in dBm

        # Plot the 2D power sweep
        if vmin is None or vmax is None:
            mesh = ax.pcolormesh(data_x, data_y, data_z, 
                                shading=style['shading'],
                                cmap=style['color_map'],
                                vmin=int(data_z.min()), vmax=int(data_z.max()))
        else:
            mesh = ax.pcolormesh(data_x, data_y, data_z,
                                shading=style['shading'],
                                cmap=style['color_map'], 
                                vmin=vmin, vmax=vmax)
        # Set the axis labels
        ax.set_xlabel("Frequency (GHz)", 
                      fontsize=style['fontsize'], 
                      labelpad=style['labelpad'])
        ax.set_ylabel("Power (dBm)", 
                      fontsize=style['fontsize'], 
                      labelpad=style['labelpad'])
        # Add a colorbar to the plot
        fig  = ax.get_figure()
        cbar = fig.colorbar(mesh, ax=ax)

        # Customize tick font size
        cbar.ax.tick_params(labelsize=style['fontsize'],
                            width=style['tick_width'],
                            length=style['tick_length'],
                            direction='out',
                            pad=style['tick_pad'])

        # Match colorbar spine width
        for spine in cbar.ax.spines.values():
            spine.set_linewidth(style['tick_width'])
        # Style label font
        cbar.set_label(z_label, 
                       fontsize=style['fontsize'], 
                       labelpad=style['labelpad'])

    # General 2D plot settings 
    if filename:
        ax.set_title(filename, 
                     fontsize=style['title_fontsize'], 
                     pad=style['title_pad'])
    # Tick parameters 
    ax.tick_params(axis='both', 
                   length=style['tick_length'], 
                   width=style['tick_width'], 
                   direction='out', 
                   labelsize=style['fontsize'], 
                   pad=style['tick_pad'])
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Spine parameters
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # Enforce y-limit boundaries
    ax.set_ylim(min(data_y), max(data_y)) 

    # Return the x, y, and z data for further processing if needed
    return data_x, data_y, data_z


# Linecut ------------------------------------------------------------------------------- #
def get_linecut(data_x, data_y, data_z, axis='x', value=None):
    if axis == 'y':
        indx = np.abs(data_y - value).argmin()
        return data_x, data_z[indx, :]
    elif axis == 'x':
        indx = np.abs(data_x - value).argmin()
        return data_y, data_z[:, indx]
def plot_linecut(ax, x, y, xlabel, ylabel, filename=None, style=None):
    ax.clear()
    # Plot the line
    ax.plot(x, y, color=style['line_color'], linewidth=style['line_width'])
    # Axes labels
    ax.set_xlabel(xlabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    ax.set_ylabel(ylabel, fontsize=style['fontsize'], labelpad=style['labelpad'])
    # Title
    if filename:
        ax.set_title(filename, fontsize=style['title_fontsize'], pad=style['title_pad'])
    # Tick styling
    ax.tick_params(
        axis='both',
        length=style['tick_length'],
        width=style['tick_width'],
        direction='in',
        labelsize=style['fontsize'],
        pad=style['tick_pad'])
    # Set major locators for x and y axes
    ax.xaxis.set_major_locator(style['x_locator'])
    ax.yaxis.set_major_locator(style['y_locator'])
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(style['tick_width'])
    # Clean y-limits to integer bounds
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(math.floor(ymin), math.ceil(ymax))

 
# Scatter Plot -------------------------------------------------------------------------- #
def scatter_1d_plot(ax, x_series, y_series, labels=None, colors=None, 
                    xlabel=None, ylabel=None, filename=None, 
                    log_y=False, ylims=None, style=None):
    ax.clear()
    # For each series in the x and y data, plot the points
    for i in range(len(x_series)):
        ax.plot(x_series[i], y_series[i], 'o',
                label=labels[i] if labels else None,
                markersize=style['marker_size'],
                color=colors[i] if colors else style['line_color'])
    # Set the general plot settings
    general_plt(ax, xlabel=xlabel, ylabel=ylabel,
                title=filename, log_y=False, ylims=None,
                style=style, legend=False)


# Multivariable Plot -------------------------------------------------------------------- #

def plot_exp_sets(ax, plot_data, xlabel=None, ylabel=None, filename=None,
                  log_y=False, ylims=None, style=None):
    # Clear the axis before plotting
    ax.clear()
    # Plot each set of data
    for entry in plot_data:
        x_val = entry['x']
        y_val = entry['y']
        label = entry.get('label', None)
        color = entry.get('color', style['line_color'])
        marks = entry.get('marks', 'o') # Default to circle marker
        # Make a plot with the correct marker style
        if marks == 'o':
            ax.plot(x_val, y_val, 'o', label=label, 
                    markersize=style['marker_size'], 
                    color=color)
        elif marks == '-':
            ax.plot(x_val, y_val, '-', label=label, 
                    linewidth=style['line_width'], 
                    color=color)
        else:
            print(f"Problem: marks must be 'o' or '-', got {marks}")
            return
    # Set the general plot settings
    general_plt(ax, xlabel=xlabel, ylabel=ylabel,
                title=filename, log_y=False, ylims=None,
                style=style, legend=True)
    return
















###########################################################################################
## Fitting Code ------------------------------------------------------------------------ ##
###########################################################################################

def linear(x, slope, intercept):
    return slope * x + intercept

def lorentzian(x, amp,center, width, offset):
    return amp*(0.5*width)**2 / ((x - center)**2 + (0.5*width)**2) + offset

def gaussian(x, amp, center, sigma, offset):
    return amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2)) + offset

# Get the appropriate fitting model and initial guess based on the fit_type
def get_fit_model(fit_type):
    models = {
        'linear':{
            'func': linear,
            'guess': lambda x, y: [np.polyfit(x, y, 1)[0], np.polyfit(x, y, 1)[1]]
        },
        'lorentzian': {
            'func': lorentzian,
            'guess': lambda x, y: [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 10, min(y)]
        },
        'gaussian': {
            'func': gaussian,
            'guess': lambda x, y: [max(y) - min(y), x[np.argmax(y)], (max(x) - min(x)) / 10, min(y)]
        }
    }
    if fit_type not in models:
        raise ValueError(f"Unsupported fit_type: {fit_type}")
    return models[fit_type]['func'], models[fit_type]['guess']

# Fit data using curve_fit from scipy.optimize
def fit_data(x, y, fit_type):
    model_func, guess_func = get_fit_model(fit_type)
    p0 = guess_func(x, y)
    popt, pcov = curve_fit(model_func, x, y, p0=p0)
    return popt, pcov, model_func



###########################################################################################
## Saving Figure Into Data Structure --------------------------------------------------- ##
###########################################################################################

# Create a figure saving path based on the database file name and the existing figures
# in the database's folder.
def create_save_path(db_path: str, fig_type:str = 'png') -> str:
    # The folder where the database file is located
    folder = os.path.dirname(db_path)
    # os.path.basename extracts the file name from the path, while then os.path.splitext
    # removes the file extension to get just the base name of the database file.
    base_name = os.path.splitext(os.path.basename(db_path))[0]
    # Check for names in the existing figures pattern: experiment_1_Fig_1.png, etc.
    pattern = re.compile(rf"{re.escape(base_name)}_Fig_(\d+)\.{fig_type}")
    # List all files (names) in the folder that match the pattern
    existing_files = [f for f in os.listdir(folder) if pattern.match(f)]
    # Extract the numeric figure suffixes and find the highest
    existing_indices = [int(pattern.match(f).group(1)) for f in existing_files]
    # Determine the next index for the new figure file
    next_index = max(existing_indices) + 1 if existing_indices else 1
    # Create the new filename with the next index and the specified figure type
    new_filename = f"{base_name}_Fig_{next_index}.{fig_type}"
    # Join the folder path and the new filename to create the full path
    return os.path.join(folder, new_filename)























class plot_main():
    def __init__(self, filename):
        self.filename = filename
        self.sweep, self.step, self.data, self.metadata = get_data_from_sqlitedb(filename)
        
        dimension = self.oneD_or_twoD()
        print('File is a ' + dimension + ' measurement')
        if dimension == '1D':
            print('arrange_data_for_plotting() returns: sweep, signal, x_label, y_label')
        elif dimension == '2D':
            print('arrange_data_for_plotting() returns: sweep, step, signal, x_label, y_label, z_label')
    
    def return_metadata(self):
        return self.metadata

    def table_names(self):
        tablenames = get_table_names(self.filename, tab = False)
        return tablenames
    
    def column_names(self, tablename):
        col_names = get_column_names_from_table(self.filename, tablename, tab = False)
        return col_names
    
    def get_sweep_type(self):
        swp = self.column_names('table_sweep')
        if swp[0] == 'freq_range':
            return 'VNA frequency (Hz)'
        else:
            return swp[0]
                                                
    def get_data_names(self):
        name = self.column_names('table_data')
        if 'Vx' in name[2]:
            return 'Lock-in signal'
        elif 'vna' in name[2]:
            return 'VNA signal'
        else:
            return name[2]
    
    def get_vna_type(self):
        cols = self.column_names('table_data')
        if len(cols) == 4:
            if 'VNA2' in cols[3] or 'vna_y2' in cols[3]:
                return 'SMITH'
            else:
                print('i dont know!!!!!')
                return 'VNA'
        else:
            return 'VNA'
        
    def get_step_type(self):
        stp = self.column_names('table_step')
        if stp == 'None':
            return None
        else:
            return stp[0]
    
    def oneD_or_twoD(self):
        stp = self.get_step_type()
        if stp == 'None':
            return '1D'
        else:
            return '2D'
        
    def if_transport_VNA(self, swp = None):
        if swp == None:
            swp = self.column_names('table_sweep')
            if 'freq_range' in swp:
                return 'VNA'
            else:
                return 'Transport'
        else:
            return swp
        
    def old_data(self, meta = True):
        if meta:
            return self.sweep, self.step, self.data, self.metadata
        else:
            return self.sweep, self.step, self.data
        
    def arrange_data_for_plotting(self, avg_len = 5, in1 = 0, in2 = 0, sig_index = 2, stype = None, **kwargs):
        swtype = self.if_transport_VNA(swp = stype)
        if swtype == 'Transport':
            if self.oneD_or_twoD() == '1D':
                print('1D transport... returning sweep, signal, x_label, y_label')
                sweep = self.sweep[:,0]
                offset_x = np.average(self.data[0:avg_len, 2])
                offset_y = np.average(self.data[0:avg_len, 3])
                signal = (np.sqrt((self.data[:, 2] - offset_x)**2 + (self.data[:, 3] - offset_y)**2))*1e3
                y_label = self.get_data_names()
                x_label = self.get_sweep_type()
                return sweep, signal, x_label, y_label

            elif self.oneD_or_twoD() == '2D':
                print('2D transport... returning sweep, step, signal, x_label, y_label, z_label')
                sweep = self.sweep[:,0]
                step = self.step[:,0]
                num_y = len(self.step[:,0])
                num_x = len(self.sweep[:,0])
                signal = np.zeros((num_y, num_x))
                for i in range(num_y):
                    data_x = self.data[i*num_x:((i+1)*num_x), 2]
                    data_y = self.data[i*num_x:((i+1)*num_x), 3]
                    offset_x = np.average(self.data[i*num_x + 0:i*num_x + 8, 2])
                    offset_y = np.average(self.data[i*num_x + 0:i*num_x + 8, 3])
                    signal[i, :] = np.sqrt((data_x - offset_x)**2 + (data_y - offset_y)**2)*1e3  # in mV
                y_label = self.get_step_type()
                x_label = self.get_sweep_type()
                z_label = self.get_data_names()
                return sweep, step, signal, x_label, y_label, z_label

        elif swtype == "VNA":
            vnatype = self.get_vna_type()
            if vnatype == 'VNA':
                if self.oneD_or_twoD() == '1D':
                    print('1D VNA... returning sweep, signal, x_label, y_label')
                    sweep = self.sweep[:,0]
                    signal = self.data[:, sig_index]
                    y_label = self.get_data_names()
                    x_label = self.get_sweep_type()
                    return sweep, signal, x_label, y_label

                elif self.oneD_or_twoD() == '2D':
                    print('2D VNA... returning sweep, step, signal, x_label, y_label, z_label')
                    sweep = self.sweep[:,0]
                    step = self.step[in1:, in2]
                    num_y = len(step)
                    num_x = len(sweep)
                    signal = np.zeros((num_y, num_x))
                    for i in range(num_y):
                        data_x = self.data[i*num_x:((i+1)*num_x), sig_index]
                        signal[i, :] = data_x
                    y_label = self.get_step_type()
                    x_label = self.get_sweep_type()
                    z_label = self.get_data_names()
                    return sweep, step, signal, x_label, y_label, z_label
            else:
                if self.oneD_or_twoD() == '1D':
                    print('VNA data is in 1D with two y-vals format... returning sweep, signal1, signal2, xlabel, y_label1, ylabel2')
                    sweep = self.sweep[:,0]
                    signal1 = self.data[:, sig_index]
                    signal2 = self.data[:, sig_index + 1]
                    y_label1 = 'vna_y1'
                    y_label2 = 'vna_y2'
                    x_label = self.get_sweep_type()
                    return sweep, signal1, signal2, x_label, y_label1, y_label2
                
                elif self.oneD_or_twoD() == '2D':
                    print('2D VNA... returning sweep, step, signal1, signal2, x_label, y_label, z_label')
                    sweep = self.sweep[:,0]
                    step = self.step[in1:, in2]
                    num_y = len(step)
                    num_x = len(sweep)
                    signal1 = np.zeros((num_y, num_x))
                    signal2 = np.zeros((num_y, num_x))
                    for i in range(num_y):
                        data_x1 = self.data[i*num_x:((i+1)*num_x), sig_index]
                        data_x2 = self.data[i*num_x:((i+1)*num_x), sig_index + 1]
                        signal1[i, :] = data_x1
                        signal2[i, :] = data_x2
                    y_label = self.get_step_type()
                    x_label = self.get_sweep_type()
                    z_label = self.get_data_names()
                    return sweep, step, signal1, signal2, x_label, y_label, z_label

class live_plot():
    def __init__(self, sweep_in:str, read_in:str, sweep_dat, data_dat, step_in = None, step_dat = None):
        self.sweep_in = sweep_in # input will be sweep_instr from experiment
        self.step_in = step_in # input will be step_instr from experiment
        self.read_in = read_in # input will be read_keys from experiment

        self.sweep_dat = sweep_dat # input will be in sweep table from experiment
        self.step_dat = step_dat # input will be in step table from experiment
        self.data_dat = data_dat # input will be in data table from experiment

        if self.step_in != None:
            self.plot_2D()

        else:
            self.plot_1D()

    def oneD_or_twoD(self):
        stp = self.step_dat
        if stp is None:
            return '1D'
        else:
            return '2D'


    def arrange_data_for_plotting(self, avg_len = 1):
        swtype = self.sweep_in
        if swtype == 'transport':
            if self.oneD_or_twoD() == '1D':
                sweep = self.sweep_dat[:,0]
                offset_x = np.average(self.data_dat[0:avg_len, 2])
                offset_y = np.average(self.data_dat[0:avg_len, 3])
                signal = (np.sqrt((self.data_dat[:, 2] - offset_x)**2 + (self.data_dat[:, 3] - offset_y)**2))*1e3
                y_label = self.read_in
                x_label = self.sweep_in
                return sweep, signal, x_label, y_label

            elif self.oneD_or_twoD() == '2D':
                print('2D transport... returning sweep, step, signal, x_label, y_label, z_label')
                sweep = self.sweep_dat[:,0]
                try:
                    step = self.step_dat[:,0]
                except:
                    step = self.step_dat
                num_y = len(step)
                num_x = len(sweep)
                signal = np.zeros((num_y, num_x))
                for i in range(num_y):
                    data_x = self.data_dat[i*num_x:((i+1)*num_x), 2]
                    data_y = self.data_dat[i*num_x:((i+1)*num_x), 3]
                    offset_x = np.average(self.data_dat[i*num_x + 0:i*num_x + 8, 2])
                    offset_y = np.average(self.data_dat[i*num_x + 0:i*num_x + 8, 3])
                    signal[i, :] = np.sqrt((data_x - offset_x)**2 + (data_y - offset_y)**2)*1e3  # in mV
                y_label = self.step_in
                x_label = self.sweep_in
                z_label = self.read_in
                return sweep, step, signal, x_label, y_label, z_label

        elif swtype == "CPWR":
            if self.oneD_or_twoD() == '1D':
                if len(self.data_dat) == 4:
                    print('VNA data is in 1D with two y-vals format... returning sweep, signal1, signal2, xlabel, y_label1, ylabel2')
                    sweep = self.sweep_dat[:,0]
                    signal1 = self.data_dat[:, 2]
                    signal2 = self.data_dat[:, 2 + 1]
                    real = signal1
                    im = signal2
                    signal = np.sqrt(real**2 + im**2)
                    y_label = 'vna_sig'
                    x_label = self.get_sweep_type()
                    return sweep, signal, x_label, y_label

                else:
                    print('1D VNA... returning sweep, signal, x_label, y_label')
                    sweep = self.sweep_dat[:,0]
                    signal = self.data_dat[:, 2]
                    y_label = self.read_in
                    x_label = self.sweep_in
                    return sweep, signal, x_label, y_label

            elif self.oneD_or_twoD() == '2D':
                if len(self.data_dat) == 4:
                    #print('2D VNA... returning sweep, step, signal, x_label, y_label, z_label')
                    sweep = self.sweep_dat[:,0]
                    try:
                        step = self.step_dat[:,0]
                    except:
                        step = self.step_dat
                    num_y = len(step)
                    num_x = len(sweep)
                    signal = np.zeros((num_y, num_x))
                    for i in range(num_y):
                        real = self.data_dat[i*num_x:((i+1)*num_x), 2]
                        im = self.data_dat[i*num_x:((i+1)*num_x), 2 + 1]
                        signal[i, :] = np.sqrt(real**2 + im**2)
                    y_label = self.get_step_type()
                    x_label = self.get_sweep_type()
                    z_label = self.get_data_names()
                    return sweep, step, signal, x_label, y_label, z_label


                else:
                    #print('2D VNA... returning sweep, step, signal, x_label, y_label, z_label')
                    sweep = self.sweep_dat[:,0]
                    try:
                        step = self.step_dat[:,0]
                    except:
                        step = self.step_dat
                    num_y = len(step)
                    num_x = len(sweep)
                    signal = np.zeros((num_y, num_x))
                    for i in range(num_y):
                        data_x = self.data_dat[i*num_x:((i+1)*num_x), 2]
                        signal[i, :] = data_x
                    y_label = self.step_in
                    x_label = self.sweep_in
                    z_label = self.read_in
                    return sweep, step, signal, x_label, y_label, z_label
            

    def plot_1D(self):
        sweep, signal, x_label, y_label = self.arrange_data_for_plotting()
        plt.plot(sweep, signal, '.-')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()

    def plot_2D(self):
        sweep, step, signal, x_label, y_label, z_label = self.arrange_data_for_plotting()
        plt.pcolormesh(sweep, step, signal)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.colorbar(label = z_label)
        plt.show()








'''
This is where I am going to put new stuff before I pass through the code above
to make a new version of the plotting code.





'''        