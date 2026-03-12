# Overview
We are ACVLAB. This is our implemented solution for "Automated Crop Disease Diagnosis from Hyperspectral Imagery 5rd" @ ICPR 2026. 

# 1. Environment
Run the prepare.sh to auto create the virtual environment:
```bash
bash -i prepare_env.sh
```

> ⚠️ The `i` in ``bash -i prepare_env.sh`` is necessary.

# 2. Download dataset & weights
## 2.1 Dataset
Download the dataset from [kaggle](https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026).

Then unzip the ```beyond-visible-spectrum-ai-for-agriculture-2026.zip```.





The folder structure will like this:
```

Dataset/
└── Kaggle_Prepared/
    ├── train/
    │   ├── HS/
    │   ├── MS/
    │   └── RGB/
    └── val/
        ├── HS/
        ├── MS/
        └── RGB/
```


# 3. Predict
Run the ```Methods.py``` can generate the submission files, will be saved in folder ```results```.

```
ICPR2026_BeyondAI_Crop_Disease/
├── Dataset/
├── output/     <- *Submissions files will saved in here*
└── Methods.py
```
