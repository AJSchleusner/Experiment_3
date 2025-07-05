import numpy as np
from newinstruments.BlueFors import BlueFors
import matplotlib.pyplot as plt

from helpers.database2 import get_table_names
from helpers.database2 import get_column_names_from_table
from helpers.database2 import get_data_from_sqlitedb


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