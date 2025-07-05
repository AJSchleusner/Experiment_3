# Experiment 3

(Project: May 2025 - )

This is the code and dependencies for my (Austin's) work involving Dr Camille Mikolas'
microchannels coupled to coplanar waveguides device.  This device was fabricated in the
University of Chicago cleanroom by LHQS alumn Dr Heejun Byeon.  Camille graduated after
conducting initial experiments on this device with an experimental setup that was not 
optimized for quantum-limited experiments.  Since then, I have added the attenuation,
amplification, and fridge-lines necessary to allow operation of Camille's device at a
much higher sensitivity.

The naming convention of 'Experiment 3' follows from Experiment 1 being an asortment of
earlier code made by Camille and Dr Niyaz Beysengulov for electrons on helium
experiments.  The subsequent 'Experiment 2' code refers to my modified versions of the 
Experiment 1 code for conducting the Bragg-Cherenkov effect experiment that I most 
recently worked on.  I will likely upload a repository of the Experiment 2 code for 
future reference. 


## Components 

The general structure of the Experiment 3 code can be broken into five interconnected 
components:

1) `experiment_v3.py`
    The experiment file is used to operate the experiment instruments for both setting 
    of sweep parameters as well as data acquisition.  This file is the most similar to
    prior code, but was largely restructured when I was creating my own version such
    that I understood what the code was doing.  Like the amateur that I am, I did
    consult with Microsoft Copilot for structure and syntax help in creating this file.

2) `acquisition_v3.ipynb`
    This is the front-facing jupyter notebook used for operating the experiment_v3.py
    file.  Through use of this file and the experiment_v3.py, data is collected and 
    saved to SQLite database files.  These are organized by their dates of creation.

3) `plotting_v3.py`
    This file is used to cleanly access the SQLite database files and create basic 
    plots of the data.  One such python file has been used to do this plotting in the 
    past by Niyaz and Camille, but I aim to format things a little differently more to
    my taste with this version of the plotting code.  In particular, I plan to change
    how the meta-data is presented to align with my previous NV-center setup.  One 
    important advantage of having the plotting separate from the acquisition notebook 
    is that this method of calling from the saved database files necessitates the data
    being saved correctly before plotting occurs.


## Components I Will Add

4) `analysis_v3.ipynb`
    Like that of the acquisition file being the front-face of the experiment_v3 code,
    this file is that front-face of the plotting_v3 code.  Here is where I will format
    plots of the database files.  It is also possible that I conduct some simulation
    in this python notebook.

5) `drivers/`
    The last component of this code being fully useable is adding the drivers called 
    by the other files.  Most are available through pymeasure, but some such as that
    of the vna are local and will be added here.
