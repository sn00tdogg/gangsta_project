# gangsta_project
## How to install requirements
Create a environment
```bash
conda create -n "name_of_environment" python=3.6
activate "name_of_environment"
pip install -r requirements.txt
```
### Folder structure required to run the project without any modifications:

```bash
|---- datasets
|     |----- chars74k-lite
|            |---- a
|            |---- b
|            |...
|            |---- z
|     |----- detection images
|---- models
|---- feature_extraction
|---- character_classification.py
|---- character_detection.py
|---- plots.py
|---- load_data.py
|---- model_weights.hdf5
```
