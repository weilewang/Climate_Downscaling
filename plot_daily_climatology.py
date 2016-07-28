#!/usr/bin/env python
import os
import sys 
import getopt
import numpy 
from pylab import *
from matplotlib import ticker

rc('xtick.major', size=6)
rc('xtick.minor', size=3)
rc('ytick.major', size=6)
rc('ytick.minor', size=3)

clr_1 = '#338800'             # green
clr_2 = '#ff5511'             # red
clr_3 = '#4477bb'             # blue
clr_4 = '#dddd11'             # yellow
clr_gray = '#888888'
clr_dark = '#666666'



IDX_LIST = {
'doy'    : 0,
'pr'     : 1,
'tasmax' : 2,
'tasmin' : 3
}
fin_tmpl = './local_day_climatology_%s.csv'
var_type = "float32"
NMETA = 5   # lines of the meta data in the in put data 
DOY = 365   # using a regular 365 calendar

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   longOpts=[ "help", "data_code=", "scenario=", "min_tas=", "max_tas=", 'min_pr=', 'max_pr=', "data_code2=", "scenario2=" ]

   min_t = 20
   max_t = 40
   min_p = 0
   max_p = 16 
   data_code1 = 'gcm_bc'
   data_scenario1 = 'none'    # not used in this version
   is_data2 = True 
   data_code2 = 'obs'
   data_scenario2 = 'none'    # not used in this version

   try:
      opts, args = getopt.getopt(sys.argv[1:], "hc:s:", longOpts)
      for o, v in opts:
         if o == "-h" or o == '--help':
            print "./dump_localclimate_ts.py --data_code  --scenario --lat_1 --lat_2 --lon_1 --lon_2"
            sys.exit(0)
         elif o == '--data_code' or o == '-c':
            data_code1 = v.strip()
         elif o == '--scenario' or o == '-s':
            data_scenario1 = v.strip()
         elif o == '--min_tas': 
            min_t = float(v)
         elif o == '--max_tas': 
            max_t = float(v)
         elif o == '--min_pr': 
            min_p = float(v)
         elif o == '--max_pr': 
            max_p = float(v)
         elif o == '--data_code2':
            data_code2 = v.strip()
            is_data2 = True
         elif o == '--scenario2':
            data_scenario2 = v.strip()
         #fi
      # end_for
   except(getopt.GetoptError), e:
      print("Warning: Unrecognizable Inputs! Stop!")
      sys.exit(-1)


   # load the daily data 
   fin = fin_tmpl % (data_code1)
   print "Processing " + fin
   data_1 = numpy.loadtxt(fin, dtype=var_type, delimiter=',', skiprows=NMETA)
 
   if is_data2 : 
      fin = fin_tmpl % (data_code2)
      print "Processing " + fin
      data2 = numpy.loadtxt(fin, dtype=var_type, delimiter=',', skiprows=NMETA)
   #fi

   figure(figsize=(8.5, 9))
   data = data_1
   # temperature
   var = 'tasmax'
   rect = [0.1, 0.56, 0.8, 0.4]
   ax_1 = axes(rect)
   ax = ax_1
   time = data[:, IDX_LIST['doy']]
   ts = data[:, IDX_LIST[var]]
   l1, = plot(time, ts, linewidth=2, linestyle='-', color=clr_1)
   majorlocator = ticker.MultipleLocator(30)
   minorlocator = ticker.MultipleLocator(2)
   ax.xaxis.set_major_locator(majorlocator)
   ax.xaxis.set_minor_locator(minorlocator)
   xlim(1, 365)
   xlabel(ur'Julian Date', fontsize=12)
   
   majorlocator = ticker.MultipleLocator(5)
   minorlocator = ticker.MultipleLocator(1)
   ax.yaxis.set_major_locator(majorlocator)
   ax.yaxis.set_minor_locator(minorlocator)
   ylim(min_t, max_t)
   ylabel('Maximum Air Temperature (K)', fontsize=12)

   # precipitation 
   var = 'pr'
   rect = [0.1, 0.1, 0.8, 0.4]
   ax_2 = axes(rect)
   ax = ax_2
   time = data[:, IDX_LIST['doy']]
   ts = data[:, IDX_LIST[var]]
   l3, = plot(time, ts, linewidth=2, linestyle='-', color=clr_1)
   majorlocator = ticker.MultipleLocator(30)
   minorlocator = ticker.MultipleLocator(2)
   ax.xaxis.set_major_locator(majorlocator)
   ax.xaxis.set_minor_locator(minorlocator)
   xlim(1, 365)
   xlabel(ur'Julian Date', fontsize=12)
   
   majorlocator = ticker.MultipleLocator(1)
   minorlocator = ticker.MultipleLocator(0.1)
   ax.yaxis.set_major_locator(majorlocator)
   ax.yaxis.set_minor_locator(minorlocator)
   ylim(min_p, max_p)
   ylabel('Precipitation (mm/day)', fontsize=12)

   if not is_data2:
      sca(ax_1)
      legend((l1, ), (data_code1,), frameon=True)
      sca(ax_2)
      legend((l3, ), (data_code1,), frameon=True)

      fimg = "loc_climatology_%s.png" % (data_code1) 
   else:
      data = data2
      var = 'tasmax'
      sca(ax_1)
      time = data[:, IDX_LIST['doy']]
      ts = data[:, IDX_LIST[var]]
      l2, = plot(time, ts, linewidth=2, linestyle='-', color=clr_2)
      legend((l1, l2), (data_code1, data_code2), frameon=True)

      sca(ax_2)
      var = 'pr'
      time = data[:, IDX_LIST['doy']]
      ts = data[:, IDX_LIST[var]]
      l4, = plot(time, ts, linewidth=2, linestyle='-', color=clr_2)
      legend((l1, l2), (data_code1, data_code2), frameon=True)
      
      fimg = "loc_climatology_%s_%s.png" % (data_code1, data_code2) 
   #fi

   savefig(fimg, dpi=100)
   show()

#fi: __main__
