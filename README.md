# VigilanceNet

Code for the ACM MM 2022 paper: "VigilanceNet: Decouple Intra- and Inter-Modality Learning for Multimodal Vigilance Estimation in RSVP-Based BCI"

---
![alt text](Model.png "VigilanceNet - DL model estimating vigilance levels using multimodal EEG and EOG")

## Install
```bash
pip install -r requirements.txt
```

### Requirements
python (3.8.11)\
torch (1.9.0)\
numpy (1.19.5)\
matplotlib (3.4.2)\
scipy (1.6.2)\
timm (0.4.13)


### Datasets
We used publicly available [SEED-VIG dataset](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)
>- 23 trials, ~ two hours EEG signal/trial
>- 17 electrode channels, 200Hz sampling rate
>- This dataset is labeled by [PERCLOS level](https://iopscience.iop.org/article/10.1088/1741-2552/aa5a98/meta?casa_token=zMmqflOHEYYAAAAA:F7YusFzBVULbjWBmoy39cvGI9RPMrUrDIOF_s1azdKrH1L0KJW9Cw_NuqFspM5OsRjMpECCpwtne)


## Train
```bash
python train.py
```


## Evaluation
```bash
python test.py
```
