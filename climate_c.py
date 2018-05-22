from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 4  # previous svg hatch linewidth
from os.path import join


def MakeDataframe(temp, prec):
    df = pd.read_csv(temp, sep=";", decimal=",")
    dfprec = pd.read_csv(prec, sep=";", decimal=",")
    df['precipitation'] = dfprec.mm
    df.drop(df.columns[[0,1,6]], axis=1, inplace=True)
    df = df.rename(columns={'Datum': 'date', 'tas':'tempMean','tasmin':'tempMin','tasmax': 'tempMax'})
    return df


class ClimateChart:
    
    def __init__(self, dataframe=None, climateChart=None):
        
        if dataframe is not None and climateChart is None:
            self.data = dataframe
            
            self.data['date'] = pd.to_datetime(self.data['date'], format='%Y-%m-%d', errors='coerce')
            self.data = self.data.set_index('date')
            self.data['tempMean'] = pd.to_numeric(self.data['tempMean'])
            self.data['tempMin'] = pd.to_numeric(self.data['tempMin'])
            self.data['tempMax'] = pd.to_numeric(self.data['tempMax'])
            
            self.temperature = self.data.groupby([(self.data.index.month)]).mean()
            self.precipitation = self.data.precipitation.groupby([(self.data.index.year),(self.data.index.month)]).sum()
            self.precipitation = self.precipitation.groupby(level=[1]).mean()
        
        elif dataframe is None and climateChart is not None:
            self.temperature = climateChart[0]
            self.precipitation = climateChart[1]
        
        else:
            print("Error!")
    
    def GetTemperature(self):
        return self.temperature.copy()
    
    def GetPrecipitation(self):
        return self.precipitation.copy()
        
    def Plot(self, title, subtitle, fontsize, filepath = None):
        # Create Canvas
#        fig = plt.figure(frameon=True)
#        fig.set_size_inches(10,6)
#        ax = plt.Axes(fig, [0,0,1,1])
#        fig.add_axes(ax)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,6), subplot_kw={'adjustable': 'box-forced'})
        fig.dpi = 75
        fig.suptitle(title , size=fontsize+4, weight='bold', horizontalalignment="center")
        
        
        # Define x-axsis
        labels = list(["J","F","M","A","M","J","J","A","S","O","N","D"])
        
        linewidth = 4
        
        # Plot Temperature
        self.tempMean_scaled = [x if x < 50 else self.scale(x, (50, 100), (50,75)) for x in self.temperature.tempMean]
        self.tempMin_scaled = [x if x < 50 else self.scale(x, (50, 100), (50,75)) for x in self.temperature.tempMin]
        self.tempMax_scaled = [x if x < 50 else self.scale(x, (50, 100), (50,75)) for x in self.temperature.tempMax]
        self.tempMean_scaled = np.array(self.tempMean_scaled)
        
        ax.plot(self.tempMean_scaled, linewidth=linewidth, color="#fdae61", label="Mean Temp.")
        ax.plot(self.tempMin_scaled, linewidth=linewidth, color='#1a9850', label="Min. Temp.")
        ax.plot(self.tempMax_scaled, linewidth=linewidth, color='#b2182b', label="Max. Temp.")

        # Set Temperature X-Axis
        ax.set_xticks(np.arange(0, 12, 1))
        ax.set_xticklabels(labels)
        
        
        
        self.precipitation_scaled_for_fill = [self.scale(x, (0, 99), (0,49)) if x < 100 else self.scale(x, (100, 600), (50,75)) for x in self.precipitation]
        #Cut off Data
        self.precipitation_scaled_for_fill = [x if x < 75 else 75 for x in self.precipitation_scaled_for_fill]
        self.precipitation_scaled_for_fill = np.array(self.precipitation_scaled_for_fill)
        
        
        
        
        ax.fill_between(np.arange(0,12,1), self.tempMean_scaled, \
                        self.precipitation_scaled_for_fill, \
                        hatch = "|", edgecolor= "#FFFFFF", facecolor="#2166ac", linewidth=0, \
                        where= self.tempMean_scaled <= self.precipitation_scaled_for_fill, interpolate=True, label="Humid")
        
        ax.fill_between(np.arange(0,12,1), self.tempMean_scaled, \
                        self.precipitation_scaled_for_fill, \
                        hatch = ".", edgecolor= "#e34a33", facecolor="#FFFFFF", linewidth=0, \
                        where= self.tempMean_scaled >= self.precipitation_scaled_for_fill, interpolate=True, label="Arid")
        
        # Format Temperature Axis
        ax.set_ylabel("Temperatur in °c", fontsize = fontsize)
        ax.set_xlabel('Monate', fontsize = fontsize)
        ax.set_yticks([-30, -20, -10, 0, 10, 20, 30, 40, 50, 55, 60, 65, 70 ,75])
        ax.set_yticklabels(["-30","-20","-10","0","10","20","30","40","50","","","","",""])
        
        for tick in ax.xaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize) 
        for tick in ax.yaxis.get_major_ticks():
                        tick.label.set_fontsize(fontsize) 
                        
        ax.set_title(subtitle, y = 1, fontsize=fontsize)
        
        
        # Make a new axis for Precipitation
        ax2 = ax.twinx()
        
        self.precipitation_scaled = [x if x < 100 else self.scale(x, (100, 600), (100,150)) for x in self.precipitation]
        self.precipitation_scaled = [x if x < 150 else 150 for x in self.precipitation_scaled]
        print(self.precipitation_scaled)
        #ax2.bar(range(0,12,1), self.precipitation_scaled, color='#2166ac')
        ax2.plot(self.precipitation_scaled, linewidth=linewidth, color="#2166ac", label="Precipitation")
        
        # Set X-Axis
        ax2.set_xticks(np.arange(0,12,1))
        
        
        ax2.set_yticks([-60, -40, -20, 0,20,40,60,80,100,110,120,130,140,150])
        ax2.set_yticklabels(["","","","0","20","40","60","80","100","200","300","400","500","600"])
        
        # Set drawing order
        ax.set_zorder(ax2.get_zorder()+1) # put ax in front of ax2
        ax.patch.set_visible(False) # hide the 'canvas' 
        
        # Format second axis
        ax2.set_ylabel('Niederschlag in mm',  fontsize = fontsize)
        ax2.tick_params(labelsize=fontsize)

        self.align_yaxis(ax, 10, ax2, 20)

        ax.set_xlim(0,11)
        
        fig.legend(loc="center", mode="expand", ncol=6, bbox_to_anchor=(0.06, -0.13, .8,.3), shadow=False, frameon=False)

        plt.show()
        
        if filepath is not None:
            fig.savefig(join(filepath, '.png'), bbox_inches=('tight'))

        

    def scale(self, val, src, dst):
        """
        Scale the given value from the scale of src to the scale of dst.
        """
        return ((val - src[0]) / (src[1]-src[0])) * (dst[1]-dst[0]) + dst[0]
    
    def align_yaxis(self, ax1, v1, ax2, v2):
        """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
        _, y1 = ax1.transData.transform((0, v1))
        _, y2 = ax2.transData.transform((0, v2))
        inv = ax2.transData.inverted()
        _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
        miny, maxy = ax2.get_ylim()
        ax2.set_ylim(miny+dy, maxy+dy)
        
    def __sub__(self, other):
        newTemperature = self.temperature - other.GetTemperature()
        newPrecipitation = self.precipitation - other.GetPrecipitation()
        
        return ClimateChart(dataframe=None, climateChart=(newTemperature, newPrecipitation)) 
    
    
