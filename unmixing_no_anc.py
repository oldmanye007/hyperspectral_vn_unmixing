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

import json
import os
import warnings
import sys
import ray
import numpy as np
#import math

sys.path.append('/home/ye6/hytools_v1/hytools-0f/')

import hytools as ht
from hytools.io.envi import *
from hytools.masks import mask_dict

warnings.filterwarnings("ignore")

def main():

    config_file = sys.argv[1]

    with open(config_file, 'r') as outfile:
        config_dict = json.load(outfile)

    images= config_dict["input_files"]

    if ray.is_initialized():
        ray.shutdown()
    print("Using %s CPUs." % config_dict['num_cpus'])
    ray.init(num_cpus = config_dict['num_cpus'])

    HyTools = ray.remote(ht.HyTools)
    actors = [HyTools.remote() for image in images]

    # Load data
    if config_dict['file_type'] == 'envi':        
        anc_files = config_dict["anc_files"]
        #pass
        _ = ray.get([a.read_file.remote(image,config_dict['file_type']) for a,image in zip(actors,images)])
    elif config_dict['file_type'] == 'neon':
        _ = ray.get([a.read_file.remote(image,config_dict['file_type']) for a,image in zip(actors,images)])

    
    for trait in config_dict['fraction_cover_models']:
        with open(trait, 'r') as json_file:
            fc_model = json.load(json_file)
            if 'sunlit' in fc_model['model']['endmembers']:
                print("Estimating fractional cover of %s endmembers:" % len(fc_model['model']['endmembers'])-1)
            else:     
                print("Estimating fractional cover of %s endmembers:" % len(fc_model['model']['endmembers']))
            for em_model in fc_model['model']['endmembers']:
                print("\t %s" % em_model["name"])


    _ = ray.get([a.do.remote(apply_fc_models,config_dict) for a in actors])
    ray.shutdown()

def apply_fc_models(hy_obj,config_dict):
    '''Apply fractional cover model(s) to image and export to file.
    '''

    hy_obj.create_bad_bands(config_dict['bad_bands'])
    hy_obj.corrections  = config_dict['corrections']

    # Load correction coefficients
    if 'topo' in  hy_obj.corrections:
        hy_obj.load_coeffs(config_dict['topo'][hy_obj.file_name],'topo')

    if 'brdf' in hy_obj.corrections:
        hy_obj.load_coeffs(config_dict['brdf'][hy_obj.file_name],'brdf')

    hy_obj.resampler['type'] = config_dict["resampling"]['type']

    for trait in config_dict['fraction_cover_models']:
        with open(trait, 'r') as json_file:
            fc_model = json.load(json_file)
            n_em = len(fc_model['model']['endmembers'])
            
            name_list=[]
            for i_em in range(n_em):
                coeffs_em = np.array(fc_model['model']['endmembers'][i_em]['coefficients'])[None,:]
                if i_em==0:
                    coeffs=coeffs_em
                else:
                    coeffs=np.vstack((coeffs, coeffs_em))
                name_list += [fc_model['model']['endmembers'][i_em]['name']]
            #intercept = np.array(trait_model['model']['intercepts'])
            model_waves = np.array(fc_model['wavelengths'])
            unit_sum_flag=config_dict['unitsum']
            

        #Check if wavelengths match
        resample = not all(x in hy_obj.wavelengths for x in model_waves)

        if resample:
            hy_obj.resampler['out_waves'] = model_waves
            hy_obj.resampler['out_fwhm'] = fc_model['fwhm']
        else:
            wave_mask = [np.argwhere(x==hy_obj.wavelengths)[0][0] for x in model_waves]

        # Build trait image file
        header_dict = hy_obj.get_header()
        header_dict['wavelength'] = []
        header_dict['data ignore value'] = -9999
        header_dict['data type'] = 4
        header_dict['band names'] = name_list + ['Sum','Mask']
        header_dict['bands'] = len(header_dict['band names'] ) 


        output_name = config_dict['output_dir']
        output_name += os.path.splitext(os.path.basename(hy_obj.file_name))[0] + "_%s" % fc_model["name"]

        writer = WriteENVI(output_name,header_dict)

        if config_dict['file_type'] == 'envi':
            iterator = hy_obj.iterate(by = 'chunk',
                      chunk_size = (64,hy_obj.columns), 
                      corrections =  hy_obj.corrections,
                      resample=resample)
        elif config_dict['file_type'] == 'neon':
            iterator = hy_obj.iterate(by = 'chunk',
                      chunk_size = (int(np.ceil(hy_obj.lines/32)),int(np.ceil(hy_obj.columns/32))), #hy_obj.columns  32,32, math.ceil(hy_obj.lines/32),math.ceil(hy_obj.columns/32)  #math.ceil(hy_obj.lines/32)//2,math.ceil(hy_obj.columns/32)//2
                      corrections =  hy_obj.corrections,
                      resample=resample)

        while not iterator.complete:
            chunk = iterator.read_next()
            if not resample:
                chunk = chunk[:,:,wave_mask]

            trait_est = np.zeros((chunk.shape[0],
                                    chunk.shape[1],
                                    header_dict['bands']))

            # Apply spectrum transforms
            for transform in  fc_model['model']["transform"]:
                if  transform== "vector":    #vnorm
                    norm = np.linalg.norm(chunk,axis=2)
                    chunk = chunk/norm[:,:,np.newaxis]
                if transform == "absorb":
                    chunk = np.log(1/chunk)
                if transform == "mean":
                    mean = chunk.mean(axis=2)
                    chunk = chunk/mean[:,:,np.newaxis]

            trait_pred = np.einsum('jkl,ml->jkm',chunk,coeffs, optimize='optimal')
         

            abnormal_mask = np.zeros((chunk.shape[0],chunk.shape[1])).astype(np.int8)

            mask_range = trait_pred > fc_model["model_diagnostics"]['min'] & \
                         (trait_pred < fc_model["model_diagnostics"]['max'])

            frac_sum=0             

            for i_em in range(n_em):
                abnormal_mask+=(2**(i_em+1)) * (1-mask_range[:,:,i_em])  

                if unit_sum_flag==True:
                    if name_list[i_em]!='sunlit':
                        frac_sum+=np.maximum(0,trait_pred[:,:,i_em])

            if unit_sum_flag==True:
                if not 'sunlit' in name_list:
                    trait_est[:,:,:n_em] = trait_pred / frac_sum[:,:,None]
                else:
                    trait_est[:,:,:n_em-1] = trait_pred[:,:,:-1] / frac_sum[:,:,None]
                    trait_est[:,:,n_em-1] = trait_pred[:,:,-1]

            else:
                trait_est[:,:,:n_em] = trait_pred

            trait_est[:,:,n_em+1] = abnormal_mask
            trait_est[:,:,n_em] = frac_sum

            nd_mask = hy_obj.mask['no_data'][iterator.current_line:iterator.current_line+chunk.shape[0],
                                             iterator.current_column:iterator.current_column+chunk.shape[1]]
            trait_est[~nd_mask] = -9999
            writer.write_chunk(trait_est,
                               iterator.current_line,
                               iterator.current_column)

            #if iterator.current_line>50: 
            #    break 

        writer.close()
        #break

if __name__== "__main__":
    main()
