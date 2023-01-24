#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Zhiwei Ye
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""


'''Template script for generating unmixing_model_estimate configuration JSON files.
'''

import os, sys
import json
import glob
import numpy as np
import pandas as pd

import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)

home = os.path.expanduser("~")

#Output path for configuration file
config_file = sys.argv[1]  #"{}/unmixing_config_workflow_avc_soap_teak_1718.json"
out_unmix_model_json = sys.argv[2]  #'{}/unmix_models/workflow_SVDIems3_vnorm_full_gen.json'.format(home)

spec_10nm_csv = "./sample_data/f170607t01p00r16_rfl_v1g_img_topo_brdf_solarMaskPCv_SVDIems3_nodark.csv"
#"/home/ye6/neon_test/unmix/f170607t01p00r16_rfl_v1g_img_topo_brdf_solarMaskPCv_SVDIems3_nodark.csv"

resample_multispec = False
multispec_dict={"s2a":"./multispec_response/sentinel2_s2a_vswir_11b.csv","OLI":"./multispec_response/OLI_vswir_7b.csv"}
multi_sensor='s2a' # OLI
multispec_response_func_csv = multispec_dict[multi_sensor]

in_images = sys.argv[3]  #"{}/AVIRIS_SOAP_2018_solar4_warp_v2"
output_dir = sys.argv[4]  #'{}/'.format(home)

spec_mode=1  #0 #1
spec_mode_range = ['vnir','full'] # 0 and 1

bad_range = [[300,400],[1320,1430],[1800,1960],[2450,2600]]  #[[300,400],[1337,1430],[1800,1960],[2450,2600]]

vecnorm_mode = True #False
pixel_applied_vn = False 
# default is false, good for sunlit fraction, no influence on fractions from VN_unmixing, good for sum of EMs from no_VN unmixing, not good for sum of EMs from VN unmixing
#if True, only good for test sum of EMs from VN unmixing
append_sunlit_frac  = True

class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)


#from https://github.com/EnSpec/hytools/blob/master/hytools/transform/resampling.py
def gaussian(x,mu,fwhm):
    """
    Args:
        x (numpy.ndarray): Values along which to generate gaussian..
        mu (float): Mean of the gaussian function..
        fwhm (float): Full width half maximum..
    Returns:
        numpy.ndarray: Gaussian along input range.
    """

    c = fwhm/(2* np.sqrt(2*np.log(2)))
    return np.exp(-1*((x-mu)**2/(2*c**2)))

def wave_interp_coeffs(hyperspec_wavelist, hyperspec_fwhm, response_func_2d, func_wave_list_1nm):

    # func_wave_list_1nm : M bands
    # hyperspec_wavelist: N bands (M>N)
    # response_func_2d  : M input wavelengths by L output wavelengths

    wave_buffer = 10
    hyperspec_gaussian_matrix = []
    min_spectrum = min(func_wave_list_1nm.min(),hyperspec_wavelist.min())//50*50 - wave_buffer
    max_spectrum = max(func_wave_list_1nm.max(),hyperspec_wavelist.max())//50*50 + wave_buffer
    one_nm = np.arange(min_spectrum,max_spectrum+1,1)

    for wave,fwhm, in zip(hyperspec_wavelist,hyperspec_fwhm):
      a =  gaussian(one_nm,wave,fwhm)
      hyperspec_gaussian_matrix.append(np.divide(a,np.sum(a)))
    hyperspec_gaussian_matrix = np.array(hyperspec_gaussian_matrix)

    response_matrix = np.zeros((one_nm.shape[0],response_func_2d.shape[1]),dtype=np.float32)
    response_2d_subset_ind = (np.argwhere(func_wave_list_1nm>=min_spectrum)[0][0],np.argwhere(func_wave_list_1nm<=max_spectrum)[-1][0])

    response_matrix[int(func_wave_list_1nm.min()-min_spectrum):int(func_wave_list_1nm.min()-min_spectrum)+response_2d_subset_ind[1]-response_2d_subset_ind[0]+1,:] = (response_func_2d / response_func_2d.sum(axis=0))[response_2d_subset_ind[0]:response_2d_subset_ind[1]+1,:]


    inv_A = np.linalg.pinv(hyperspec_gaussian_matrix)

    transform_matrix = response_matrix.T@inv_A

    return transform_matrix 

def export_json(coeffs_array, outjson, name_str,f_wave,f_fwhm,transform_mode, endmember_name_list):
  #print("Converting to {}.\n".format(outjson))
  with open(outjson, 'w') as outfile:

      common_item["name"] = name_str
      
      common_item["wavelengths"] = f_wave.tolist()

      common_item["fwhm"] =  f_fwhm.tolist()  

      common_item['model'] = {}
      common_item['model']["components"] = 1
      common_item['model']["transform"] = transform_mode # ['vector'] #('vector','absorb','mean')
      common_item['model']["endmembers"] = []

      for i in range(coeffs_array.shape[0]):

         common_item['model']["endmembers"] += [{'name':endmember_name_list[i],'coefficients':coeffs_array[i,:].tolist()}] 
      
      model_coeffs = common_item
      
      json.dump(model_coeffs,outfile)  #, cls=DecimalEncoder  
      
      logging.info("Unmix model file is saved:{}".format(outjson))


config_dict = {}
config_dict['file_type'] = 'envi'
config_dict["output_dir"] = output_dir 

config_dict['bad_bands'] = bad_range

# Input data settings for NEON
#################################################################
# config_dict['file_type'] = 'neon'
# images= glob.glob("*.h5")
# images.sort()
# config_dict["input_files"] = images

# Input data settings for ENVI
#################################################################
''' Only differnce between ENVI and NEON settings is the specification
of the ancillary datasets (ex. viewing and solar geometry). All hytools
functions assume that the ancillary data and the image date are the same
size, spatially, and are ENVI formatted files.

The ancillary parameter is a dictionary with a key per image. Each value
per image is also a dictionary where the key is the dataset name and the
value is list consisting of the file path and the band number.
'''

config_dict['file_type'] = 'envi'
aviris_anc_names = ['path_length','sensor_az','sensor_zn',
                    'solar_az', 'solar_zn','phase','slope',
                    'aspect', 'cosine_i','utc_time']
images= glob.glob(in_images)
images.sort()
config_dict["input_files"] = images

config_dict["anc_files"] = {}

config_dict['num_cpus'] = len(images)

# Assign correction coefficients
##########################################################
''' Specify correction(s) to apply and paths to coefficients.
'''

config_dict['corrections'] = []  #['topo','brdf']
config_dict['unitsum'] = True

# Select wavelength resampling type
##########################################################
'''Wavelength resampler will only be used if image wavelengths
and model wavelengths do not match exactly

See image_correct_json_generate.py for options.

'''
config_dict["resampling"]  = {}
config_dict["resampling"]['type'] =  'cubic'

# Masks
##########################################################
'''Specify list of masking layers to be appended to the
trait map. Each will be placed in a seperate layer.

For no masks provide an empty list: []
'''
config_dict["masks"] = []

# Define trait coefficients
##########################################################
## Generate json coefficients

df_spec_10nm = pd.read_csv(spec_10nm_csv)
col_names = list(df_spec_10nm.columns)

em_name_list = col_names[1:]

if spec_mode==0:
#model_range = [1000, 2400]
    model_range = [400, 1100] # VNIR
elif spec_mode==1:
    model_range = [400, 2450]  # FULL
else:
    quit()

logging.info("Model spectral range:{}".format(model_range))    
    
in_wave_list = df_spec_10nm.iloc[:,0].astype(np.float32)

em_spec = df_spec_10nm.iloc[:,1:].to_numpy().astype(np.float32)


model_bbl  = np.ones(in_wave_list.shape[0])

model_bbl[(in_wave_list<model_range[0]) | (in_wave_list>model_range[1])]=0 
model_bbl = model_bbl.astype(np.bool)
   
if 'fwhm' in df_spec_10nm.columns:
    fwhm = df_spec_10nm[:,col_names.index('fwhm')]
else:
    fwhm = np.array([10.0]*in_wave_list.shape[0])

if resample_multispec==False:
   logging.info("Model coefficients are estimated with wavelengths resampled to {}".format(multi_sensor))   

   em_spec_vn = em_spec[model_bbl] / np.linalg.norm(em_spec[model_bbl,:],axis=0)

   out_mat = np.matmul(np.linalg.inv(em_spec[model_bbl].T@em_spec[model_bbl]),em_spec[model_bbl].T)

   sunlit_coeff = np.sum(out_mat,axis=0) # No vn is needed

   out_mat_vn = np.matmul(np.linalg.inv(em_spec_vn.T@em_spec_vn),em_spec_vn.T)

   if append_sunlit_frac:
      em_name_list+=['sunlit']


   if vecnorm_mode:
      model_name = "AllFracVNorm"+spec_mode_range[spec_mode]
      out_mat_return=out_mat_vn
   else:
      model_name = "AllFrac"+spec_mode_range[spec_mode]
      out_mat_return=out_mat

   if append_sunlit_frac:
      out_mat_return = np.vstack((out_mat_return,sunlit_coeff))

else:
   response_func = pd.read_csv(multispec_response_func_csv)
   multispec_band_names = list(response_func.columns)[1:]
   response_wavelist = response_func.iloc[:,0].to_numpy().astype(np.float32)
   func_coeff = response_func.iloc[:,1:].to_numpy().astype(np.float32)
   func_coeff[np.isnan(func_coeff)]=0

   multispec_wavelist = response_wavelist[None,:]@func_coeff/np.sum(func_coeff,axis=0)

   transform_coeffs = wave_interp_coeffs(in_wave_list[model_bbl],fwhm[model_bbl], func_coeff, response_wavelist)

   em_spec_multi = transform_coeffs@em_spec[model_bbl]

   em_spec_multi_vn = em_spec_multi / np.linalg.norm(em_spec_multi,axis=0)
   
   out_mat = np.matmul(np.linalg.inv(em_spec[model_bbl].T@em_spec[model_bbl]),em_spec[model_bbl].T)

   sunlit_coeff = np.sum(out_mat,axis=0) # No vn is needed

   out_mat_multispec = np.matmul(np.linalg.inv(em_spec_multi.T@em_spec_multi),em_spec_multi.T)
   out_mat_multispec_vn = np.matmul(np.linalg.inv(em_spec_multi_vn.T@em_spec_multi_vn),em_spec_multi_vn.T)

   if append_sunlit_frac:
      em_name_list+=['sunlit']
      
   if pixel_applied_vn:   
      vnorm_transform=['vector']
   else:
      vnorm_transform=[]
   logging.info("Vector Normalization on each pixel when applying model: {}".format(pixel_applied_vn))
   
   if vecnorm_mode:
      model_name = "AllFracVNorm"+spec_mode_range[spec_mode]+multi_sensor
      out_mat_return=out_mat_multispec_vn@transform_coeffs
   else:
      model_name = "AllFrac"+spec_mode_range[spec_mode]+multi_sensor
      out_mat_return=out_mat_multispec@transform_coeffs
      
   logging.info("Vector Normalize input Endmembers: {}".format(vecnorm_mode))

   if append_sunlit_frac:
      out_mat_return = np.vstack((out_mat_return,sunlit_coeff))

   logging.info("Append a band of SUNLIT Fraction: {}".format(append_sunlit_frac))


common_item = {"wavelength_units":"nanometers", "type":"linear_unmixing", "spectrometer":'AVC', "description":''}
    
common_item["model_diagnostics"] = {"min":0,  "max": 2, "rmse": None, "r_squared": None}#{"min":0,  "max": 400, "rmse": None, "r_squared": None}
common_item["units"] = ''

export_json(out_mat_return, out_unmix_model_json, model_name,in_wave_list[model_bbl],fwhm[model_bbl],vnorm_transform,em_name_list)
logging.info("Endmember list: {}".format(em_name_list))


###########################################################
models = glob.glob(out_unmix_model_json)
models.sort()
config_dict["fraction_cover_models"]  = models

with open(config_file, 'w') as outfile:
    json.dump(config_dict,outfile, indent=4)
    logging.info("Configuration file is saved:{}".format(config_file))

    