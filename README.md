# RGB/FLIR Lettuce Detection
This script detects individual plants in GeoTIF images, including plot clipped GeoTIFs. The detections are saved as a CSV file containing plot information and phenotype traits, such as bounding area.

## Inputs
Directory containing geoTIFFs. 

## Outputs
* CSV file containing:
    * Date
    * Plot
    * Genotype
    * Bounding box corner and center coordinates (EPSG:4326) 
    * Bounding area in m<sup>2</sup>. 

## Arguments and Flags
* **Positional Arguments:** 
    * **Directory containing files to process:** 'dir' 
* **Required Arguments:**
    * **A .pth model file:** '-m', '--model'
    * **GeoJSON containing plot boundaries:** '-g', '--geojson'
    * **Scan date:** '-d', '--date'
    * **Specify if FLIR or RGB images:** '-t', '--type', choices=['FLIR', 'RGB']                  

* **Optional Arguments:**
    * **Output directory:** '-o', '--outdir', default='detect_out'
    * **Classes to detect:** '-c', '--detect_class', nargs='+', default=['lettuce']
