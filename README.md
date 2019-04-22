# gangsta_project
## How to install requirements
Create a environment
```bash
conda create -n "name_of_environment" python=3.6
conda activate "name_of_environment"
pip install -r requirements.txt
```
### Folder structure required to run the project without any modifications:

```bash
|---- datasets  # Folder for data sets
|     |----- chars74k-lite  # Datasets requiered to train the models
|            |---- a
|            |---- b
|            |...
|            |---- z
|     |----- detection images  # Datasets requiered to do character detection
|---- models
|     |---- CNN.py
|     |---- KNN.py
|     |---- SVM.py
|     |---- Naive_Bayes.py
|---- feature_extraction
|     |---- data_processing.py
|     |---- filters.py
|---- character_classification.py
|---- character_detection.py
|---- plots.py
|---- load_data.py
|---- model_weights.hdf5
```
