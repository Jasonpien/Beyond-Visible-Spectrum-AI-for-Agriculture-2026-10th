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
Run the ```ML_methods.ipynb``` can generate the submission files, will be saved in folder ```results```.

```
D:\SIDE_PROJECT\AUTOMATED-CROP-DISEASE-DIAGNOSIS-FROM-HYPERSPECTRAL-IMAGERY-3RD
├─beyond-visible-spectrum-ai-for-agriculture-2024
│  ├─archive
│  │  ├─train
│  │  │  ├─Health
│  │  │  ├─Other
│  │  │  └─Rust
│  │  └─val
│  │      └─val
│  └─ICPR01
│      └─kaggle
│          ├─1
│          ├─2
│          └─evaluation
├─results                   <- *Submissions files will saved in here*
└─__pycache__
```
