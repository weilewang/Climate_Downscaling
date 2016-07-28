# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 13:28:30 2015

calculating annually heat wave numbers (loop for every year, from 1950 - 2099)
using a specified threshold value for tmax 

@author: alice zhang
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

IDX_LIST = {
'year'   : 0,
'month'  : 1,
'day'    : 2,
'doy'    : 3,
'tasmax' : 4,
'tasmin' : 5,
'pr'     : 6
}

TCRIT = 32     # critical Tmax
TLENS = 3      # minimum duration of a heatwave 
DOY = 365      # days of year, with leap years truncated
NMETA = 0 

#----------------------------------------------------------------------------

if __name__ == "__main__":
   var_type = "float32"
   data_code = 'gcm_bc'
   dpath = "."
   fin = '%s/local_day_clim_%s.csv' % (dpath, data_code)
   data = np.loadtxt(fin, dtype=var_type, delimiter=',', skiprows=NMETA)  
   # select tmax colum
   tmax = data[:, IDX_LIST['tasmax']]
   ianc = np.where(tmax>=TCRIT, 1, 0)

   fdata = "./annual_heatwave_numbers_%s.csv" % (data_code)
   fout = open(fdata, "w")

   year_s = int(data[0, IDX_LIST['year']])   
   year_e = int(data[-1, IDX_LIST['year']])   
   nyears = year_e - year_s + 1
   for i in range(nyears):
      idx_s = i*DOY
      idx_e = idx_s + DOY
      iszero = np.zeros(DOY+2, dtype='int8')
      iszero[1:-1] = ianc[idx_s:idx_e]
      tmax_annual = tmax[idx_s:idx_e]
      
      absdiff = np.abs(np.diff(iszero))                        # detect the changes from 0 to 1 or from 1 to 0
      ranges = np.argwhere(absdiff==1).reshape(-1, 2)          # range are in pairs, range[0] indicates the start of the warm spell
                                                               # and range[1] indicates the end of the warm spell
      ndays_spell = ranges[:,1] - ranges[:,0]                  # No need to add 1 here
      ranges = np.compress(ndays_spell>=TLENS, ranges, axis=0) # determine the heatwaves 
      number_heatwave = len(ranges)
      
      intensity_heatwave = 0
      for j in range(number_heatwave):
         intensity_heatwave += tmax_annual[ranges[j, 0]:ranges[j, 1]].sum()
      # end_for: j
      fout.write( ("%4d,%02d,%5.8e \n" % (i+data[0,0], number_heatwave, intensity_heatwave)) )

   # end_for: i
   fout.close()

   # OK, read the results from fdata for plotting
   data=np.loadtxt(fdata, dtype=var_type, delimiter=',')
   year = data[:, 0]
   hw_number = data[:, 1]
   hw_intensity = data[:, 2]
   plt.subplot(2, 1, 1)
   plt.scatter(year,hw_number)
   plt.plot(year,hw_number,"black")
   plt.ylabel("Annual Heat Wave Number")
   plt.title("Heat Wave in Data: %s" % (data_code))
   plt.xlim([year_s, year_e])

   plt.subplot(2, 1, 2)
   plt.plot(year,hw_intensity)
   plt.xlabel("Year")
   plt.ylabel("Heat Wave Intensity")
   plt.xlim([year_s, year_e])
   #plt.title("Heat Wave in New York City 1950-2099 (%s/rcp8.5)" % (gcm))
   plt.savefig("heatwave_%s.png"%(data_code))
   plt.show()
# end_if: __main__
