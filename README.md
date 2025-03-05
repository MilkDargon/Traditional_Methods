## Traffic flow prediction implemented based on traditional methods (HA, SVR, VAR)


#### Table of Contents
* configs: training Configs and model configs for each dataset
* metrics: evaluate metrics

#### Required Packages

```
numpy==1.24.4
pandas==2.0.3
PyYAML==6.0.2
scikit_learn==1.3.2
scipy==1.15.2
statsmodels==0.14.1
tqdm==4.65.2
```

#### Data Preparation
For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/file/d/1VoZgubahWLNNg7Jn1-WzPdPvc1z4Tf3H/view?usp=drive_link) or [Baidu Netdisk](https://pan.baidu.com/s/1s2psErR6Kjfl-Lxu54Nwcg?pwd=sq66 
).  
Unzip the downloaded dataset files to the main file directory.

#### Training Commands
For example: If you want to run HA.

```bash
cd Traditional_Methods-master/
python run_HA.py -d dataset
```

`dataset`:
- METRLA
- PEMSBAY
- PEMS03
- PEMS04
- PEMS07
- PEMS08

