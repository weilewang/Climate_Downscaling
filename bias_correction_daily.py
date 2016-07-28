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
'pr'     : 6
}

DEBUG = True
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
   
   return trend 
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
   return (mask, trend_full, anom_full)
# end_def: safe_detrend 

#----------------------------------------------------------------------------------------------------------------------
def calc_cdf(data, M=None, smooth=True):
  
   if M is None:
      M = len(data) / 2
   #fi: M

   pdf_x, bin_x = numpy.histogram(data, M, normed=True)     # the returned bin_x are M+1 edges
   bin_x = (bin_x[:-1] + bin_x[1:]) * 0.5                   # calculate bin_x at the centers  
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
      bin_x = (numpy.arange(len(pdf_x)) - int((nwnd+1)/2)) * bin_width + bin_x[0]
   #fi: smooth
   
   cdf_x = numpy.add.accumulate(pdf_x) * bin_width

   return (cdf_x, bin_x)
   
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

   return rslt
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

   return rslt
# end_def: cdf2val


#----------------------------------------------------------------------------------------------------------------------
def bc_cdf_match(x_obs, y_mdl, base_s, base_e, detrend=True):
  
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
   cdf_obs, bin_obs = calc_cdf(anom_obs.flatten(), (N*M)/2, True)
   cdf_mdl2, bin_mdl2 = calc_cdf(anom_mdl2.flatten(), (N*M)/2, True)

   if DEBUG:
      figure(figsize=(11, 8.5))
      ioff()
      l1, = plot(bin_obs+trend_obs.mean(), cdf_obs, 'r') 
      l2, = plot(bin_mdl2+trend_mdl2.mean(), cdf_mdl2, 'b')
      ylim(0, 1)
      ylabel('Cumulative Distribution Function (0-1)')
      xlabel('Climate Variable')
      legend((l1, l2), ('obs', 'model'), loc='best')
      show()
   # fi

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

   return anom_mdl

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

   return rslt

# end_def: calc_daily_mask

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

   # for simplicity, below we used some "hard-coded" arguments like input file names, etc.
   baseyr_s = 1981
   baseyr_e = 2014

   DOY = 365

   # 1. load the observation data
   fdata = 'local_day_clim_obs.csv'
   nmeta_obs = 1
   data = numpy.loadtxt(fdata, dtype='f4', delimiter=',', skiprows=nmeta_obs)
   year_obs  = data[:, IDX_LIST['year']].astype('i4')
   tmax_obs  = data[:, IDX_LIST['tasmax']]
   tmin_obs  = data[:, IDX_LIST['tasmin']]
   prcp_obs  = data[:, IDX_LIST['pr']]
   yr_s_obs = year_obs[0]
   yr_e_obs = year_obs[-1] 
   nyrs = yr_e_obs - yr_s_obs + 1
   tmax_obs = tmax_obs.reshape(nyrs, DOY)
   tmin_obs = tmin_obs.reshape(nyrs, DOY)
   prcp_obs = prcp_obs.reshape(nyrs, DOY)
   # adjust the data to baseline years
   indx_s = baseyr_s - yr_s_obs
   indx_e = baseyr_e - yr_s_obs + 1
   tmax_obs = tmax_obs[indx_s:indx_e, :]
   tmin_obs = tmin_obs[indx_s:indx_e, :]
   prcp_obs = prcp_obs[indx_s:indx_e, :]
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
   tmax_mdl  = data[:, IDX_LIST['tasmax']]
   tmin_mdl  = data[:, IDX_LIST['tasmin']]
   prcp_mdl  = data[:, IDX_LIST['pr']]
   data = []                              # Done with data and so the memory can be recycled.
   yr_s_mdl = year_mdl[0] 
   yr_e_mdl = year_mdl[-1] 
   nyrs = yr_e_mdl - yr_s_mdl + 1
   tmax_mdl = tmax_mdl.reshape(nyrs, DOY)
   tmin_mdl = tmin_mdl.reshape(nyrs, DOY)
   prcp_mdl = prcp_mdl.reshape(nyrs, DOY)
   # subset the data to baseline years
   indx_s = baseyr_s - yr_s_mdl
   indx_e = baseyr_e - yr_s_mdl + 1

   # 3. start the bias_correction loop
   tmax_bc = numpy.zeros_like(tmax_mdl)
   tmin_bc = numpy.zeros_like(tmin_mdl)
   prcp_bc = numpy.zeros_like(prcp_mdl)
   DEBUG_save = DEBUG
   NWIN_DAY = 30
   jday_array = numpy.arange(1, DOY+1, dtype='i4')
   for jd in range(DOY):
      if jd > 0:
         DEBUG = False        # turn-off the debug plotting for the rest of the year
      #fi

      jd_mask = calc_daily_mask(jd+1, jday_array, NWIN_DAY)
     
      # tmax
      x_obs = tmax_obs.compress(jd_mask, axis=-1)
      y_mdl = tmax_mdl.compress(jd_mask, axis=-1)
      # call the algorithm: bias correction by matching CDFs 
      rslt = bc_cdf_match(x_obs, y_mdl, indx_s, indx_e, True)  # for temperature we set detrend=True
      jday_cmp = jday_array.compress(jd_mask)
      tmax_bc[:,jd] = rslt.compress((jday_cmp==(jd+1)), axis=-1).flatten() 
      
      # tmin
      x_obs = tmin_obs.compress(jd_mask, axis=-1)
      y_mdl = tmin_mdl.compress(jd_mask, axis=-1)
      # call the algorithm: bias correction by matching CDFs 
      rslt = bc_cdf_match(x_obs, y_mdl, indx_s, indx_e, True)  # for temperature we set detrend=True
      jday_cmp = jday_array.compress(jd_mask)
      tmin_bc[:,jd] = rslt.compress((jday_cmp==(jd+1)), axis=-1).flatten() 
      
      # pr
      x_obs = prcp_obs.compress(jd_mask, axis=-1)
      y_mdl = prcp_mdl.compress(jd_mask, axis=-1)
      # call the algorithm: bias correction by matching CDFs 
      rslt = bc_cdf_match(x_obs, y_mdl, indx_s, indx_e, False) # for precipitation detrending is not necessary; you can compare the difference 
      prcp_bc[:, jd] = rslt.compress((jday_cmp==(jd+1)), axis=-1).flatten()
   # end_for:
   DEBUG = DEBUG_save

   # save the data 
   fout = './local_day_clim_gcm_bc.csv'
   rslt = numpy.column_stack((year_mdl, month_mdl, day_mdl, doy_mdl, tmax_bc.flat,tmin_bc.flat, prcp_bc.flat))
   numpy.savetxt(fout, rslt, fmt='%.6e', delimiter=',')
   

   # 4. debug: plot the time series 
   if DEBUG :
  
      tmax_bc = numpy.where(tmax_bc<>FILLVALUE, tmax_bc, nan)
      tmax_mdl = numpy.where(tmax_mdl<>FILLVALUE, tmax_mdl, nan)
      tmax_obs = numpy.where(tmax_obs<>FILLVALUE, tmax_obs, nan)
      tmax_bc = tmax_bc.mean(axis=-1)
      tmax_mdl = tmax_mdl.mean(axis=-1)
      tmax_obs = tmax_obs.mean(axis=-1)
      tmax_obs = where(tmax_obs< -50, None, tmax_obs) 

      prcp_bc = numpy.where(prcp_bc<>FILLVALUE, prcp_bc, nan)
      prcp_mdl = numpy.where(prcp_mdl<>FILLVALUE, prcp_mdl, nan)
      prcp_obs = numpy.where(prcp_obs<>FILLVALUE, prcp_obs, nan)
      prcp_bc = prcp_bc.mean(axis=-1)*365
      prcp_mdl = prcp_mdl.mean(axis=-1)*365
      prcp_obs = prcp_obs.mean(axis=-1)*365
      prcp_obs = where(prcp_obs< 0, None, prcp_obs) 

      figure(figsize=(11, 8.5))
      subplot(2, 1, 1)
      l1, = plot(range(yr_s_obs, yr_e_obs+1), tmax_obs, 'r') 
      l2, = plot(range(yr_s_mdl, yr_e_mdl+1), tmax_mdl, 'b') 
      l3, = plot(range(yr_s_mdl, yr_e_mdl+1), tmax_bc, 'g') 
      ylabel('TAVE (DegC)')
      legend((l1, l2, l3), ('obs', 'model (uncorrected)', 'model (corrected)'), loc='best')
      subplot(2, 1, 2)
      l1, = plot(range(yr_s_obs, yr_e_obs+1), prcp_obs, 'r') 
      l2, = plot(range(yr_s_mdl, yr_e_mdl+1), prcp_mdl, 'b') 
      l3, = plot(range(yr_s_mdl, yr_e_mdl+1), prcp_bc, 'g') 
      ylabel('PRCP (mm/yr)')
      xlabel('YEAR')
      legend((l1, l2, l3), ('obs', 'model (uncorrected)', 'model (corrected)'), loc='best')
      show()
   # fi: DEBUG
