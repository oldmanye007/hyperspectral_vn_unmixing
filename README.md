# hyperspectral_vn_unmixing
Scripts for a basic land cover subpixel unmixing, particularly for vegetation fraction cover estimation, using a universal method based on HyTools

The main step of the method is to vector nomalize the input endmembers with L2-norm to make each of the edmembers unit-norm. However, pixels in the input images are not required to undertake the same vector normalization due to a unit-sum post-processing.   


## Results from the workflow

Regular output: One band for each endmember, plus a band for Sum of all fractions before unit-sum post-prosessing, and a band for mask.

Mask band: They should be integer numbers (although in floating point format)
The N+1 th bit is for the N th endmember. 0 in that bit is for narmal range, and 1 is for mask that's out of the range. In a summary, 0 for the whole number is basically good number.

Optional output: Sunlit fraction band


## Script Description

The first script is to generate a configuration file that includes all the settings of the unmixing model prediction.

```bash
$indir='/input_dir/'
$outdir='/output_dir/'

out_config_file="unmixing_config_workflow_avc.json"
out_unmix_model_json="unmix_models/workflow_SVDIems3_vnorm_gen.json"

$input_images="AVIRIS_*_solar4_warp_v2"

python unmix_estimate_json_generate_avc_workflow.py "$outdir"/"$out_config_file" "$outdir"/"$out_unmix_model_json" "$indir"/"$input_images" "$outdir"
```

Default is to use all usable wavelength/bands in either the VNIR or VNIR+SWIR regions

```python
# unmix_estimate_json_generate_avc_workflow.py
spec_mode=1 # VNIR+SWIR regions
spec_mode_range = ['vnir','full']
resample_multispec = False
```
or
```python
# unmix_estimate_json_generate_avc_workflow.py
spec_mode=0 # VNIR
spec_mode_range = ['vnir','full']
resample_multispec = False
```

It is also possible to use less wavelengths by resampling to multispectral bands during model coefficients estimation.

Enable multispectral resampling using the setting of Sentinel-2 in the script
```python
# unmix_estimate_json_generate_avc_workflow.py
resample_multispec = True
multi_sensor='s2a' # OLI
```

Enable multispectral resampling using the setting of Landsat8 in the scipt
```python
# unmix_estimate_json_generate_avc_workflow.py
resample_multispec = True
multi_sensor='OLI'
```


The second script needs to be run in an environment with HyTools.

```bash
# appply the model to images
python unmixing_no_anc.py "$outdir"/"$out_config_file"
```

Run these two scripts in order.