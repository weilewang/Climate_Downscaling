#!/usr/bin/env python
import os
import sys 
import getopt
import numpy 
from pylab import *

IDX_LIST = {
'year'   : 0,
'month'  : 1,
'day'    : 2,
'doy'    : 3,
'tasmax' : 4,
'tasmin' : 5,
'pr'     : 6
}
var_list = ['pr', 'tasmax', 'tasmin']
fin_tmpl = 'local_day_clim_%s.csv' 
fout_tmpl = 'local_day_climatology_%s.csv'
var_type = "float32"
NMETA = 5   # lines of the meta data in the in put data 
DOY = 365   # using a regular 365 calendar
NWIN = 30   # window for composite daily data 
FILLVALUE = -9999.0

#----------------------------------------------------------------------------------------------------------------------
def calc_daily_mask(doy, doy_array, window):
   doy_s = doy - window/2
   doy_e = doy + window/2
   rslt = numpy.zeros(len(doy_array), dtype='bool') # zeros means "False" for boolean variables
   if doy_s < 1:
      doy_s += DOY
      rslt = numpy.where(doy_array >= doy_s, True, rslt)
      doy_s = 1  # reset doy_s to 1
   # fi
   if doy_e > DOY:
      doy_e -= DOY 
      rslt = numpy.where(doy_array <= doy_e, True, rslt)
      doy_e = DOY+1  # including possible Day 366
   # fi
   rslt = numpy.where(numpy.logical_and(doy_array>=doy_s, doy_array<=doy_e), True, rslt)
   return rslt

# end_def: calc_daily_mask

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
   longOpts=[ "help", "data_code=", "scenario=" ]
   
   # default settings
   NMETA = 0 
   data_code = 'gcm_bc'
   data_scenario = 'historical'
   
   try:
      opts, args = getopt.getopt(sys.argv[1:], "hc:s:", longOpts)
      for o, v in opts:
         if o == "-h" or o == '--help':
            print "./calc_climatology.py --data_code  --scenario "
            sys.exit(0)
         elif o == '--data_code' or o == '-c':
            data_code = v.strip()
         elif o == '--scenario' or o == '-s':
            data_scenario = v.strip()
         #fi
      # end_for
   except(getopt.GetoptError), e:
      print("Warning: Unrecognizable Inputs! Stop!")
      sys.exit(-1)

   # load the daily data 
   #fin = fin_tmpl % (data_code, data_scenario)
   fin = fin_tmpl % (data_code)
   print "Processing " + fin
   data_all = numpy.loadtxt(fin, dtype=var_type, delimiter=',', skiprows=NMETA)
   
   if data_scenario == 'historical':
      year_s = 1981
      year_e = 2014
      year_array = data_all[:, IDX_LIST['year']]
      mask = numpy.where(numpy.logical_and(year_array>=year_s, year_array<=year_e), True, False)
      data_all = numpy.compress(mask, data_all, axis=0)
   #if


   # start the loop
   NVAR = len(var_list)
   rslt_mean = numpy.zeros((DOY, NVAR), dtype=var_type)
   for j in range(DOY):
      doy = j + 1
      mask = calc_daily_mask(doy, data_all[:, IDX_LIST['doy']], NWIN)
      #DEBUG: print j, mask.sum() 
      
      data_day = numpy.compress(mask, data_all, axis=0)
      # take care of the FILLVALUES
      data_day = numpy.compress(numpy.all(data_day<>FILLVALUE, axis=1), data_day, axis=0)
      for i in range(NVAR):
         rslt_mean[j, i] = data_day[:, IDX_LIST[var_list[i]]].mean()
      # end_for: i
   # end_for: j

   # output
   fout = fout_tmpl % (data_code)
   f = open(fout, 'w')
   # copy META data from fin to fout
   f2 = open(fin, 'r')
   for i in range(NMETA-1):
      f.write( f2.readline() )
   # end_for: i
   f2.close()
   f.write("DOY")
   for ivar in range(NVAR):
      f.write( (", %s" % (var_list[ivar])) )
   #end_for
   f.write("\n")

   for i in range(DOY):
      f.write( ("%03d" % (i+1)) )
      for ivar in range(NVAR):
         f.write( (", %.6e" % (rslt_mean[i, ivar])) )
      #end_for
      f.write("\n")
   #end_for
   f.close() 

#fi: __main__
