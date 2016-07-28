#!/usr/bin/env python
import sys
import os.path
import numpy
from pylab import *

IDX_LIST = {
'year'   : 0,
'month'  : 1,
'day'    : 2,
'doy'    : 3,
'tasmax' : 4,
'tasmin' : 5,
'pr'     : 6,
}

MOY = 12
FILLVALUE = -9999.0

#----------------------------------------------------------------------------------------------------------------------
def trend_core(x, mask):
   # x : 1-d (n) array
   N = len(x)
   k = 2
   Y = ones((N, k), 'f4')
   Y[:, 1] = arange(N, dtype='f4')
   
   # compress x and Y
   x = x.compress(mask)
   Z = Y.compress(mask, axis=0)
   (p, residual, rank, s) = linalg.lstsq(Z, x)
   
   # calculate trend with uncompressed Y
   trend = dot(Y, p.reshape(k, 1)).flatten()
   
   return trend.copy() 
# end_def: trend_core


#----------------------------------------------------------------------------------------------------------------------
def safe_detrend(data, detrend=True):
   # data is a 2D array 
   mask = numpy.all( (data <> FILLVALUE), axis=-1)
   x = data.mean(axis=-1).astype('f4') 
   if detrend:
      # x : 1-d (n) array
      winsize = 15
      N = len(x)
      M = int((winsize-1)/2)*2+1    # make it an odd number
      indx_s = (M-1)/2        # zero_based starting point
      indx_e = N - (M+1)/2    # zero_based ending point
      # calculate the trend
      trend_full = numpy.ones_like(x)*FILLVALUE
      for i in range(indx_s, indx_e+1):
         i_s = i - indx_s  # indx_s also represents the half window size
         i_e = i + indx_s
         y = trend_core(x[i_s:i_e+1], mask[i_s:i_e+1])
         
         trend_full[i] = y[indx_s]  #
         if i == indx_s:               # the very beginning
            trend_full[:i] = y[:indx_s]
         elif i == indx_e:             # the end
            trend_full[i:] = y[indx_s:]
         # end_if
      # end_for: i
      
      # calculate the anomalies 
      anom_full = data.copy()
      (row, col) = data.shape
      for j in range(col):
         anom_full[:, j] -= trend_full
      # end_for: j
      anom_full = numpy.where(data<>FILLVALUE, anom_full, FILLVALUE) 
   else:
      anom_full = data.copy()
      trend_full = numpy.zeros_like(x)
   #fi
   return (mask.copy(), trend_full.copy(), anom_full.copy())
# end_def: safe_detrend 

#----------------------------------------------------------------------------------------------------------------------
def calc_cdf(data, M=None, smooth=True):
  
   if M is None:
      M = len(data) / 2
   #fi: M

   pdf_x, bin_x = numpy.histogram(data, bins=M, normed=True)     # the returned bin_x are M+1 edges
   bin_width = bin_x[1] - bin_x[0]

   if smooth:
      # smooth the PDFs with a Gaussian core
      nwnd = M / 3                                          # nwnd is a very important parameter that controls the spread of the signal
      if (nwnd % 2) == 0: nwnd += 1                         # make it an odd number
      steps = numpy.arange(nwnd, dtype='f4') - (nwnd-1)/2           
      steps = steps/steps[-1] * 3.0                         # In normalized Gaussian distribution N(3.0) approximates 0s   
      gwnd = numpy.exp(-numpy.power(steps, 2))               
      gwnd = gwnd / gwnd.sum()  # further normalize the window
      # smoothing
      pdf_x = numpy.convolve(pdf_x, gwnd)
      bin_x = (numpy.arange(len(pdf_x)+1) - int((nwnd+1)/2)) * bin_width + bin_x[0]
   #fi: smooth
   
   cdf_x = numpy.add.accumulate(pdf_x) * bin_width


   bin_center = (bin_x[:-1] + bin_x[1:]) * 0.5                   # calculate bin_x at the centers  
   bin_edge = bin_x.copy()
   return (cdf_x.copy(), bin_center, bin_edge)
   
# end_def: calc_cdf


#----------------------------------------------------------------------------------------------------------------------
def val2cdf(cdf_x, bin_x, data):
   
   M = len(cdf_x)
   bin_width = bin_x[1] - bin_x[0]

   # using the vector operation
   idx = bin_x.searchsorted(data)
   idx_pre = idx - 1
   # underflow: for simplicity, we are using cut-offs for under/over flows. But you can change this...
   idx_pre = numpy.where(idx_pre<0, 0, idx_pre)
   # overlfow:
   idx = numpy.where(idx==M, M-1, idx)
   
   rslt = cdf_x[idx_pre] + (cdf_x[idx]-cdf_x[idx_pre])/(bin_width) * (data - bin_x[idx_pre])

   return rslt.copy()
# end_def: val2cdf

#----------------------------------------------------------------------------------------------------------------------
def cdf2val(cdf_x, bin_x, data):
   
   M = len(cdf_x)
   bin_width = bin_x[1] - bin_x[0]

   # using the vector operation
   idx = cdf_x.searchsorted(data)
   idx_pre = idx - 1
   # underflow: for simplicity, we are using cut-offs for under/over-flows. 
   idx_pre = numpy.where(idx_pre<0, 0, idx_pre)
   # overlfow:
   idx = numpy.where(idx==M, M-1, idx)

   # to avoid "divide by 0" problems under the under/over-flows situations
   cdf_diff1 = (cdf_x[idx]-cdf_x[idx_pre])  
   cdf_diff2 = (data-cdf_x[idx_pre])
   mask = (cdf_diff1 < cdf_diff2) | (cdf_diff2 <= 0) | (cdf_diff1 <= 0) 
   cdf_diff1 = numpy.where(mask, 1, cdf_diff1)
   cdf_diff2 = numpy.where(mask, 0, cdf_diff2)

   rslt = bin_x[idx_pre] + bin_width * (cdf_diff2/cdf_diff1) 

   return rslt.copy()
# end_def: cdf2val


#----------------------------------------------------------------------------------------------------------------------
def bc_cdf_match(x_obs, y_mdl, base_s, base_e, detrend=True, cdf_smooth=True):
  
   # detrend the data
   (mask_obs, trend_obs, anom_obs) = safe_detrend(x_obs, detrend)
   (mask_mdl, trend_mdl, anom_mdl) = safe_detrend(y_mdl, detrend)
   mask_base = mask_mdl[base_s:base_e] & mask_obs

   anom_obs = anom_obs.compress(mask_base, axis=0)
   trend_obs = trend_obs.compress(mask_base)
   anom_mdl2 = anom_mdl[base_s:base_e].compress(mask_base, axis=0)
   trend_mdl2 = trend_mdl[base_s:base_e].compress(mask_base)

   # calculate the CDFs for the obs and the base data
   (N, M) = anom_obs.shape
   cdf_obs, bin_obs, bin_obs_e = calc_cdf(anom_obs.flatten(), (N*M)/2, cdf_smooth)
   cdf_mdl2, bin_mdl2, bin_mdl2_e = calc_cdf(anom_mdl2.flatten(), (N*M)/2, cdf_smooth)

   (N, M) = anom_mdl.shape
   # mapping anom_mdl to probabilities with CDF_mdl2
   prob_mdl = val2cdf(cdf_mdl2, bin_mdl2, anom_mdl.flatten())
   # and then re-mapping it back to corrected climate anomalies using CDF_obs
   anom_mdl = cdf2val(cdf_obs, bin_obs, prob_mdl)

   # adjust the mean of the trend in the base window
   trend_mdl += (trend_obs.mean() - trend_mdl2.mean()) 

   anom_mdl = anom_mdl.reshape(N, M)
   for i in range(M):
      anom_mdl[:, i] += trend_mdl
   # end_for

   return anom_mdl.copy()

# end_def bc_cdf_match

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

   return rslt.copy()

# end_def: calc_daily_mask

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

   # for simplicity, below we used some "hard-coded" arguments like input file names, etc.
   # we will use the period 1950 - 2006 as the baseline time window
   baseyr_s = 1950
   baseyr_e = 2010

   jday_test = 45 
   DOY = 365
   var = 'tasmax'

   # 1. load the observation data
   fdata = 'local_day_clim_obs.csv'
   nmeta_obs = 1
   data = numpy.loadtxt(fdata, dtype='f4', delimiter=',', skiprows=nmeta_obs)
   year_obs  = data[:, IDX_LIST['year']].astype('i4')
   doy_obs  = data[:, IDX_LIST['doy']].astype('i4')
   yr_s_obs = year_obs[0]
   yr_e_obs = year_obs[-1] 
   nyrs = yr_e_obs - yr_s_obs + 1
   clm_obs  = data[:, IDX_LIST[var]]
   clm_obs = clm_obs.reshape(nyrs, DOY)
   year_obs = year_obs.reshape(nyrs, DOY)
   doy_obs = doy_obs.reshape(nyrs, DOY)
   # adjust the data to baseline years
   indx_s = baseyr_s - yr_s_obs
   indx_e = baseyr_e - yr_s_obs + 1
   clm_obs = clm_obs[indx_s:indx_e, :]
   year_obs = year_obs[indx_s:indx_e, :].flatten()
   doy_obs = doy_obs[indx_s:indx_e, :].flatten()
   yr_s_obs = baseyr_s
   yr_e_obs = baseyr_e

   # 2. load the model data
   fdata = 'local_day_clim_gcm.csv'
   nmeta_gcm = 5
   data = numpy.loadtxt(fdata, dtype='f4', delimiter=',', skiprows=nmeta_gcm)
   year_mdl  = data[:, IDX_LIST['year']].astype('i4')
   month_mdl  = data[:, IDX_LIST['month']].astype('i4')
   day_mdl  = data[:, IDX_LIST['day']].astype('i4')
   doy_mdl  = data[:, IDX_LIST['doy']].astype('i4')
   clm_mdl  = data[:, IDX_LIST[var]]
   data = []                              # Done with data and so the memory can be recycled.
   yr_s_mdl = year_mdl[0] 
   yr_e_mdl = year_mdl[-1] 
   nyrs = yr_e_mdl - yr_s_mdl + 1
   clm_mdl = clm_mdl.reshape(nyrs, DOY)
   # subset the data to baseline years
   indx_s = baseyr_s - yr_s_mdl
   indx_e = baseyr_e - yr_s_mdl + 1
   clm_mdl = clm_mdl[indx_s:indx_e]
   indx_e -= indx_s
   indx_s -= indx_s

   # 3. process the data for the specified jday 
   NWIN_DAY = 40
   jday_array = numpy.arange(1, DOY+1, dtype='i4')
   
   jd = jday_test 
   jd_mask = calc_daily_mask(jd, jday_array, NWIN_DAY)

   x_obs = clm_obs.compress(jd_mask, axis=-1)
   y_mdl = clm_mdl.compress(jd_mask, axis=-1)
   
   if var.lower() == 'pr':
      detrend = False
   else:
      detrend = True
   #fi
   y_bc = bc_cdf_match(x_obs, y_mdl, indx_s, indx_e, detrend) 


   # for plotting, flatten the data and filter out fill values
   # we assume that the data are all truncated to baseline years
   x_obs = x_obs.flatten()
   y_mdl = y_mdl.flatten()
   y_bc = y_bc.flatten()
   mask = (x_obs <> FILLVALUE) & (y_mdl <> FILLVALUE) & (y_bc <> FILLVALUE)
   x_obs = x_obs.compress(mask)
   y_mdl = y_mdl.compress(mask)
   y_bc = y_bc.compress(mask)

   # plot the scatter plot
   l1, = plot(x_obs, y_mdl, 'bo')
   l2, = plot(x_obs, y_bc, 'mo')
   v_min = min(x_obs.min(), y_mdl.min())
   v_max = max(x_obs.max(), y_mdl.max())
   xlim(v_min, v_max)
   ylim(v_min, v_max)
   plot([v_min, v_max], [v_min, v_max], 'k--')
   xlabel('Observed ' + var)
   ylabel('Simulated ' + var)
   legend((l1,l2,), ('before BC', 'after BC'), loc='best')
   title('Sample Data for Julian Day: ' + str(jday_test))
   show()

   # plot the histogram
   pdf_obs, bin_obs = numpy.histogram(x_obs, bins=50, normed=True)     # the returned bin_x are M+1 edges
   pdf_mdl, bin_mdl = numpy.histogram(y_mdl, bins=bin_obs, normed=True)
   pdf_bc, bin_bc = numpy.histogram(y_bc, bins=bin_obs, normed=True)
   bin_width = bin_obs[1] - bin_obs[0]
   bin_ctr = 0.5*(bin_obs[0:-1]+bin_obs[1:])
   subplot(2, 1, 1)
   l1,=plot(bin_ctr, pdf_obs*bin_width, 'r-o')
   l2,=plot(bin_ctr, pdf_mdl*bin_width, 'g-o')
   l3,=plot(bin_ctr, pdf_bc*bin_width, 'b-o')
   ylabel('pdf')
   legend((l1, l2, l3), ('obs', 'mdl', 'bc'))
#   legend((l1, l2,), ('obs', 'mdl',))

   subplot(2, 1, 2)
   plot(bin_ctr, add.accumulate(pdf_obs)*bin_width, 'r-o')
   plot(bin_ctr, add.accumulate(pdf_mdl)*bin_width, 'g-o')
   plot(bin_ctr, add.accumulate(pdf_bc)*bin_width, 'b-o')
   ylabel('cdf')
   xlabel(var + " of Julian Day: " + str(jday_test))
   show()
