# FLiR Lettuce Detection

## Inputs


## Outputs

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
       
## Adapting the Script
                                        
### Example
