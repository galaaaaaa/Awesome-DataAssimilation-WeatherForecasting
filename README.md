# Awesome Data Assimilation and Weather Forecasting

papers and codes on data assimilation and weather forecasting based on AI methods, and their commonly used datasets.
## Table of Contents

[Data Assimilation](#data-assimilation)

[Weather Forecasting](#weather-forecasting)

## Data Assimilation

### 2025

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| VAE-Var: Variational-Autoencoder-Enhanced Variational Assimilation [PDF](https://arxiv.org/abs/2405.13711) | ICLR 2025 | Lorenz 63、Lorenz 96 | - |
| LATENT-ENSF: A LATENT ENSEMBLE SCORE FILTER FOR HIGH-DIMENSIONAL DATA ASSIMILATION WITH SPARSE OBSERVATION DATA [PDF](https://arxiv.org/abs/2409.00127) | ICLR 2025 | SWE、ERA5 | - |
|Deep Bayesian Filter for Bayes-faithful Data Assimilation [PDF](https://arxiv.org/abs/2405.18674)|ICML 2025|moving MNIST、double pendulum、Lorenz96| - |
|Tensor-Var: Efficient Four-Dimensional Variational Data Assimilation [PDF](https://arxiv.org/abs/2501.13312)|ICML 2025|WeatherBench2|[Code](https://github.com/yyimingucl/TensorVar)|
|FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations [PDF](https://www.nature.com/articles/s41612-025-01039-3)|nature npj 2025|ERA5|[Code](https://github.com/xuxiaoze/FuXi-DA)|
|Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales [PDF](https://arxiv.org/abs/2406.16947)|JAMES 2025|HRRR、ISD|-|


### 2024

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| FNP: Fourier Neural Processes for Arbitrary-Resolution Data Assimilation [PDF](https://papers.nips.cc/paper_files/paper/2024/file/f93d03f2ad836c815b7ca60dfbe23bf8-Paper-Conference.pdf) | NIPS 2024 | ERA5 |  [Code](https://github.com/OpenEarthLab/FNP) |
| On conditional diffusion models for PDE simulations [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2974844555dc383ea16c5f35833c7a57-Paper-Conference.pdf) | NIPS 2024 | Kuramoto-Sivashinsky、Kolmogorov、Burgers  | [Code](https://github.com/cambridge-mlg/pdediff) |
| Fuxi-DA: A Generalized Deep Learning Data Assimilation Framework for Assimilating Satellite Observations [PDF](https://arxiv.org/abs/2404.08522) | -- | ERA5 | - |

### 2023

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| Score-based Data Assimilation [PDF](https://papers.nips.cc/paper_files/paper/2023/file/7f7fa581cc8a1970a4332920cdf87395-Paper-Conference.pdf) | NIPS 2023 | - | [Code](https://github.com/francois-rozet/sda) |
| Machine Learning With Data Assimilation and Uncertainty Quantification for Dynamical Systems: A Review [PDF](https://arxiv.org/pdf/2303.10462) | IEEE/CAA Journal of Automatica Sinica | - | - |
| Data Assimilation using ERA5, ASOS, and the U-STN model for Weather Forecasting over the UK [PDF](https://arxiv.org/pdf/2401.07604) | NIPS 2023 | ERA5、ASOS | [Code](https://github.com/acse-ww721/DA_ML_ERA5_ASOS_Weather_Forecasting_UK) |


### Earlier

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| Learning to Assimilate in Chaotic Dynamical Systems [PDF](https://proceedings.neurips.cc/paper_files/paper/2021/file/65cc2c8205a05d7379fa3a6386f710e1-Paper.pdf) [PDF- Supplemental](https://papers.neurips.cc/paper_files/paper/2021/file/65cc2c8205a05d7379fa3a6386f710e1-Supplemental.pdf) | NIPS 2021 | - | [Code](https://github.com/mikemccabe210/amortizedassimilation) |
|Integrating data assimilation with structurally equivariant spatial transformers: Physically consistent data-driven models for weather forecasting [PDF](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_37.pdf) | NIPS 2020 | ERA5 |-|
| Completing physics-based models by learning hidden dynamics through data assimilation [PDF](https://ai4earthscience.github.io/neurips-2020-workshop/papers/ai4earth_neurips_2020_30.pdf) | NIPS 2020 | - | - |
| Towards physically consistent data-driven weather forecasting: Integrating data assimilation with equivariance-preserving deep spatial transformers [PDF](https://gmd.copernicus.org/articles/15/2221/2022/gmd-15-2221-2022.pdf) | Geoscientific Model Development(2022) | ERA5 | [Code](https://github.com/ashesh6810/DDWP-DA) |

## Weather Forecasting

### 2025

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| HR-Extreme: A High-Resolution Dataset for Extreme Weather Forecasting [PDF](https://arxiv.org/abs/2409.18885) | ICLR 2025 | - | [CODE](https://github.com/HuskyNian/HR-Extreme) |
| Continuous Ensemble Weather Forecasting with Diffusion models [PDF](https://arxiv.org/abs/2410.05431) | ICLR 2025 | - | [CODE](https://github.com/martinandrae/Continuous-Ensemble-Forecasting) |
|Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function[PDF](https://arxiv.org/abs/2501.19374#) | ICML 2025 | HRES | - |
|OneForecast: A Universal Framework for Global and Regional Weather Forecasting[PDF](https://arxiv.org/abs/2502.00338) | ICML 2025 | weatherbench | [CODE](https://github.com/YuanGao-YG/OneForecast) |



### 2024

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| CLIMODE: CLIMATE AND WEATHER FORECASTING WITH PHYSICS-INFORMED NEURAL ODES [PDF](https://arxiv.org/abs/2404.10024) | ICLR 2024 | ERA5 | [CODE](https://github.com/Aalto-QuML/ClimODE) |
| Generalizing Weather Forecast to Fine-grained Temporal Scales via Physics-AI Hybrid Modeling [PDF](https://arxiv.org/abs/2405.13796) | NeurIPS 2024 | ERA5 | [CODE](https://github.com/black-yt/WeatherGFT) |
| WeatherBench 2: A benchmark for the next generation of data-driven global weather models[PDF](https://arxiv.org/pdf/2308.15560) | Journal of Advances in Modeling Earth Systems | several: [Intro.](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)、[Google Cloud bucket](https://console.cloud.google.com/storage/browser/weatherbench2) | [Code](https://github.com/google-research/weatherbench2)、[official website](https://sites.research.google/weatherbench/) |


### 2023

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
|FuXi: A cascade machine learning forecasting system for 15-day global weather forecast [PDF](https://arxiv.org/abs/2306.12873) | npj 2023 | ERA5、HREs-dc0、ENS-fc0 | [CODE](https://github.com/tpys/FuXi)  |

### Earlier

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|

