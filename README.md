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
| Deep Bayesian Filter for Bayes-faithful Data Assimilation [PDF](https://arxiv.org/abs/2405.18674)|ICML 2025|moving MNIST、double pendulum、Lorenz96| - |
| Tensor-Var: Efficient Four-Dimensional Variational Data Assimilation [PDF](https://arxiv.org/abs/2501.13312)|ICML 2025|WeatherBench2|[Code](https://github.com/yyimingucl/TensorVar)|
| FuXi-DA: a generalized deep learning data assimilation framework for assimilating satellite observations [PDF](https://www.nature.com/articles/s41612-025-01039-3)|nature npj 2025|ERA5|[CODE](https://github.com/xuxiaoze/FuXi-DA)|


### 2024

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| FNP: Fourier Neural Processes for Arbitrary-Resolution Data Assimilation [PDF](https://papers.nips.cc/paper_files/paper/2024/file/f93d03f2ad836c815b7ca60dfbe23bf8-Paper-Conference.pdf) | NIPS 2024 | ERA5 |  [Code](https://github.com/OpenEarthLab/FNP) |
| On conditional diffusion models for PDE simulations [PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/2974844555dc383ea16c5f35833c7a57-Paper-Conference.pdf) | NIPS 2024 | Kuramoto-Sivashinsky、Kolmogorov、Burgers  | [Code](https://github.com/cambridge-mlg/pdediff) |
| Fuxi-DA: A Generalized Deep Learning Data Assimilation Framework for Assimilating Satellite Observations [PDF](https://arxiv.org/abs/2404.08522) | -- | ERA5 | - |
| Generative Data Assimilation of Sparse Weather Station Observations at Kilometer Scales [PDF](https://arxiv.org/abs/2406.16947)|JAMES 2024|HRRR、ISD|-|
| FengWu-4DVar: Coupling the Data-driven Weather Forecasting Model with 4D Variational Assimilation [PDF](https://arxiv.org/abs/2312.12455) | ICML 2024 | ERA5  | [Code](https://github.com/OpenEarthLab/FengWu-4DVar) |
| (U-STN) Towards physics-inspired data-driven weather forecasting: integrating data assimilation with a deep spatial-transformer-based u-net in a case study with EAR5[PDF](https://gmd.copernicus.org/articles/15/2221/2022/) | Geoscientific Model Development 2024 | ERA5 | [Code](https://zenodo.org/records/6112374) |

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
| A foundation model for the Earth system [PDF](https://www.nature.com/articles/s41586-025-09005-y) | Nature | HRES/ERA5/... | [CODE](https://microsoft.github.io/aurora/intro.html) |
| LaDCast: A Latent Diffusion Model for Medium-Range Ensemble Weather Forecasting [PDF](https://arxiv.org/abs/2506.09193) | ArXiv | ERA5 | [CODE](https://github.com/tonyzyl/ladcast) |
| HR-Extreme: A High-Resolution Dataset for Extreme Weather Forecasting [PDF](https://arxiv.org/abs/2409.18885) | ICLR 2025 | - | [CODE](https://github.com/HuskyNian/HR-Extreme) |
| Continuous Ensemble Weather Forecasting with Diffusion models [PDF](https://arxiv.org/abs/2410.05431) | ICLR 2025 | - | [CODE](https://github.com/martinandrae/Continuous-Ensemble-Forecasting) |
|Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function[PDF](https://arxiv.org/abs/2501.19374#) | ICML 2025 | HRES | - |
|OneForecast: A Universal Framework for Global and Regional Weather Forecasting[PDF](https://arxiv.org/abs/2502.00338) | ICML 2025 | weatherbench | [CODE](https://github.com/YuanGao-YG/OneForecast) |
|An artificial intelligence-based limited area model for forecasting of surface meteorological variables[PDF](https://www.nature.com/articles/s43247-025-02347-5) | (Nature 子刊)communications earth & environment | HRRR/HadISD | [CODE](https://github.com/PaddlePaddle/PaddleScience/tree/develop/examples/yinglong) |
| Physics-Guided Learning of Meteorological Dynamics[PDF](https://arxiv.org/abs/2505.14555) | KKD | HRES/ERA5/WeatherBench | [Code](https://github.com/yingtaoluo/PhyDL-NWP) |
| End-to-end data-driven weather prediction[PDF](https://www.nature.com/articles/s41586-025-08897-0) | Nature | ERA5 | [Code](https://github.com/annavaughan/aardvark-weather-public) |
| The operational medium-range deterministic weather forecasting can be extended beyond a 10-day lead time[PDF](https://www.nature.com/articles/s43247-025-02502-y) | Nature | ERA5 | [Code](https://github.com/OpenEarthLab/FengWu) |
| (TianXing) linear complexity transformer model with explicit attention decay for global weather forecasting[PDF](https://link.springer.com/article/10.1007/s00376-024-3313-9) | Advances in Atmospheric Sciences | ERA5 |-|


### 2024

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| CLIMODE: CLIMATE AND WEATHER FORECASTING WITH PHYSICS-INFORMED NEURAL ODES [PDF](https://arxiv.org/abs/2404.10024) | ICLR 2024 | ERA5 | [CODE](https://github.com/Aalto-QuML/ClimODE) |
| Generalizing Weather Forecast to Fine-grained Temporal Scales via Physics-AI Hybrid Modeling [PDF](https://arxiv.org/abs/2405.13796) | NeurIPS 2024 | ERA5 | [CODE](https://github.com/black-yt/WeatherGFT) |
| WeatherBench 2: A benchmark for the next generation of data-driven global weather models[PDF](https://arxiv.org/pdf/2308.15560) | Journal of Advances in Modeling Earth Systems | several: [Intro.](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)、[Google Cloud bucket](https://console.cloud.google.com/storage/browser/weatherbench2) | [Code](https://github.com/google-research/weatherbench2)、[official website](https://sites.research.google/weatherbench/) |
| Scaling transformer neural networks for skillful  and reliable medium-range weather forecasting [PDF](https://arxiv.org/abs/2312.03876)|NIPS 2024|WeatherBench 2|[CODE](https://github.com/tung-nd/stormer)|
| HEAL-ViT: Vision Transformers on a spherical mesh for medium-range weather forecasting [PDF](https://arxiv.org/abs/2403.17016)|arXiv 2024|ERA5|-|
| (PFformer) PFformer: A Time-Series Forecasting Model for Short-Term Precipitation Forecasting[PDF](https://ieeexplore.ieee.org/abstract/document/10678751) | IEEE Access 2024 | https://rp5.ru/ | - |
| (OMG-HD) OMG-HD: A high-resolution ai weather model for end-to-end forecasts from observations[PDF](https://arxiv.org/pdf/2412.18239) | arXiv 2024 | RTMA | - |
| AIFS -- ECMWF's data-driven forecasting system [PDF](https://arxiv.org/abs/2406.01465)|arXiv 2024|ERA5|-|
| (GraphDOP) GraphDOP: Towards skilful data-driven medium-range weather forecasts learnt and initialised directly from observations [PDF](https://www.arxiv.org/abs/2412.15687)|arXiv 2024|观测空间原始观测|-|
| (WeatherODE) Mitigating Time Discretization Challenges with WeatherODE: A Sandwich Physics-Driven Neural ODE for Weather Forecasting [PDF](https://arxiv.org/abs/2410.06560)|arXiv 2024|ERA5|[CODE](https://github.com/DAMO-DI-ML/WeatherODE)|
| (NeuralGCM) Neural general circulation models for weather and climate [PDF](https://www.nature.com/articles/s41586-024-07744-y)|Nature 2024|ERA5|[CODE](https://github.com/neuralgcm/neuralgcm)|
| (Conformer) STC-ViT: Spatio Temporal Continuous Vision Transformer for Weather Forecasting [PDF](https://arxiv.org/abs/2402.17966)|arXiv 2024|ERA5|-|
| (Prithvi WxC) Prithvi WxC: Foundation Model for Weather and Climate [PDF](https://arxiv.org/abs/2409.13598)|arXiv 2024|NASA MERRA-2、CORDEX-EUR-11、WeatherBench-2|[CODE](https://github.com/NASA-IMPACT/Prithvi-WxC)|
| (AtmosArena) AtmosArena: Benchmarking Foundation Models for Atmospheric Sciences [PDF](https://openreview.net/forum?id=cUucUH9y0s)|NeurIPS 2024 Workshop FM4Science|ERA5|[CODE](https://github.com/tung-nd/atmos-arena?tab=readme-ov-file)|
| (ω-GNN) Coupling Physical Factors for Precipitation Forecast in China With Graph Neural Network [PDF](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106676) | AGU 2024 | ERA5、CMPA | - |
| (WeatherGNN) WeatherGNN: Exploiting Meteo- and Spatial-Dependencies for Local Numerical Weather Prediction Bias-Correction[PDF](https://www.ijcai.org/proceedings/2024/0269.pdf) | IJCAI 2024 | Ningbo、Ningxia | - |
| (MPNNs) Multi-modal graph neural networks for localized off-grid weather forecasting [PDF1](https://arxiv.org/abs/2410.12938) [PDF2](https://openreview.net/pdf?id=CN328Aw03P) | arXiv 2024 | ERA5、MADIS | [Code](https://github.com/Earth-Intelligence-Lab/LocalizedWeatherGNN/) |
| (MetMamba) MetMamba: Regional Weather Forecasting with Spatial-Temporal Mamba Model[PDF](https://arxiv.org/abs/2408.06400) | arXiv 2024 | ERA5 | - |
| (MambaDS) MambaDS: Near-Surface Meteorological Field Downscaling With Topography Constrained Selective State-Space Modeling[PDF](https://arxiv.org/abs/2408.10854#:~:text=In%20this%20paper%2C%20we%20address%20these%20limitations%20by,downscaling%20and%20propose%20a%20novel%20model%20called%20MambaDS.) | IEEE Transactions on Geoscience and Remote Sensing 2024 | ERA5 | - |
| (DeepPhysiNet) DeepPhysiNet: Bridging Deep Learning and Atmospheric Physics for Accurate and Continuous Weather Modeling[PDF](https://arxiv.org/abs/2401.04125) | arXiv 2024 | NCEP IFS、ERA5、Weather2K | [Code](https://github.com/flyakon/DeepPhysiNet) |
| CasCast: Skillful High-resolution Precipitation Nowcasting via Cascaded Modelling[PDF](https://arxiv.org/abs/2402.04290) | ICML 2024 | - | [Code](https://github.com/OpenEarthLab/CasCast) |
| SRNDiff: Short-term Rainfall Nowcasting with Condition Diffusion Model[PDF](https://arxiv.org/abs/2402.13737) | arXiv 2024 | - | [Code](https://github.com/ybu-lxd/SRNDiff) |
| DiffCast: A Unified Framework via Residual Diffusion for Precipitation Nowcasting[PDF](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_DiffCast_A_Unified_Framework_via_Residual_Diffusion_for_Precipitation_Nowcasting_CVPR_2024_paper.html) | CVPR 2024 | - | [Code](https://github.com/DeminYu98/DiffCast) |
| CoDiCast: Conditional Diffusion Model for Weather Prediction with Uncertainty Quantification[PDF](https://arxiv.org/abs/2409.05975) | arXiv 2024 | - | [Code](https://github.com/JimengShi/CoDiCast) |
| Continuous Ensemble Weather Forecasting with Diffusion models[PDF](https://arxiv.org/abs/2410.05431) | arXiv 2024 | - | [Code](https://github.com/martinandrae/Continuous-Ensemble-Forecasting) |
| GenCast: Diffusion-based ensemble forecasting for medium-range weather[PDF](https://www.nature.com/articles/s41586-024-08252-9) | Nature 2024 | - | [Code](https://github.com/google-deepmind/graphcast) |
| SEEDs: Emulation of Weather Forecast Ensembles with Diffusion Models[PDF](https://www.science.org/doi/10.1126/sciadv.adk4489) | Science Advances 2024 | - | [Code](https://github.com/google-research/google-research/tree/master/seeds) |


### 2023

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
|FuXi: A cascade machine learning forecasting system for 15-day global weather forecast [PDF](https://arxiv.org/abs/2306.12873) | npj 2023 | ERA5、HREs-dc0、ENS-fc0 | [CODE](https://github.com/tpys/FuXi)  |
| FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators[PDF](https://dl.acm.org/doi/10.1145/3592979.3593412) | ICLR 2023 | ERA5 | [CODE](https://github.com/NVlabs/FourCastNet) |
| SwinVRNN: A Data-Driven Ensemble Forecasting Model via Learned Distribution Perturbation[PDF](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022MS003211) | JAMES | WeatherBench | [CODE](https://github.com/tpys/wwprediction) |
| SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting [PDF](https://ojs.aaai.org/index.php/AAAI/article/view/25105) | AAAI 2023 | ERA5 |-|
| (Pangu-Weather) Accurate medium-range global weather forecasting with 3D neural networks[PDF](https://www.nature.com/articles/s41586-023-06185-3) | Nature 2023 | ERA5 | [CODE](https://github.com/198808xc/Pangu-Weather) |
| (GraphCast) Graphcast: Learning skillful medium-range global weather forecasting[PDF](https://www.science.org/doi/10.1126/science.adi2336) | Science 2023 | ERA5 | [CODE](https://github.com/openclimatefix/graph_weather) |
| (ClimaX) ClimaX: A foundation model for weather and climate [PDF](https://arxiv.org/abs/2301.10343)|ICML 2023|ERA5|[CODE](https://github.com/microsoft/ClimaX)|
| W-MAE: Pre-trained weather model with masked autoencoder for multi-variable weather forecasting [PDF](https://arxiv.org/abs/2304.08754)|arXiv 2023|ERA5|[CODE](https://github.com/Gufrannn/W-MAE)|
| (HiSTGNN) HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting Information[PDF](https://www.sciencedirect.com/science/article/abs/pii/S0020025523011659) | Information Sciences 2023 | WD | [Code](https://github.com/mb-Ma/HiSTGNN) |
| (MetNet-3) Deep Learning for Day Forecasts from Sparse Observations[PDF](https://arxiv.org/abs/2306.06079) | arXiv 2023 | - | - |
| (PredRNN) PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning[PDF](https://ieeexplore.ieee.org/abstract/document/9749915) | IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE 2023 | Moving MNIST、KTH | [Code](https://github.com/thuml/predrnn-pytorch) |
| (MM-RNN) MM-RNN: A Multimodal RNN for Precipitation Nowcasting[PDF](https://ieeexplore.ieee.org/abstract/document/10092888) | IEEE Transactions on Geoscience and Remote Sensing 2023 | MeteoNet、RAIN-F | - |
| (NowcastNet) Skilful nowcasting of extreme precipitation with NowcastNet[PDF](https://www.nature.com/articles/s41586-023-06184-4) | Nature 2023 | MRMS | [Code](https://codeocean.com/capsule/3935105/tree/v1) |
| Latent diffusion models for generative precipitation nowcasting with accurate uncertainty quantification[PDF](https://arxiv.org/abs/2304.12891) | arXiv 2023 | - | [Code](https://github.com/MeteoSwiss/ldcast) |
| PreDiff: Precipitation Nowcasting with Latent Diffusion Models[PDF](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html) | NeurIPS 2023 | - | [Code](https://proceedings.neurips.cc/paper_files/paper/2023/file/f82ba6a6b981fbbecf5f2ee5de7db39c-Supplemental-Conference.zip) |
| Precipitation nowcasting with generative diffusion models[PDF](https://arxiv.org/abs/2308.06733) | arXiv 2023 | - | [Code](https://github.com/fmerizzi/Precipitation-nowcasting-with-generative-diffusion-models) |
| STGM: Physical-Dynamic-Driven AI-Synthetic Precipitation Nowcasting Using Task-Segmented Generative Model[PDF](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023GL106084) | AGU 2023 | - | [Code](https://zenodo.org/records/8380856) |
| PCT-CycleGAN: Paired Complementary Temporal Cycle-Consistent Adversarial Networks for Radar-Based Precipitation Nowcasting[PDF](https://dl.acm.org/doi/abs/10.1145/3583780.3615006) | ACM 2023 | - | - |

### Earlier

| Title | Venue | Dataset | CODE |
|:-------|:-------:|:---------:|:------:|
| (GnnWeather) Forecasting Global Weather with Graph Neural Networks[PDF](https://arxiv.org/abs/2202.07575) | arXiv 2022 | ERA5 | [CODE](https://github.com/rkeisler/keisler22-predict?tab=readme-ov-file) |
| (SwinUnet)Spatiotemporal Vision Transformer for Short Time Weather Forecasting[PDF](https://sci-hub.se/10.1109/BigData52589.2021.9671442) | IEEE BigData 2021 | Weather4cast | [Code](https://github.com/bojesomo/Weather4Cast2021-SwinUNet3D) |
| (Earthformer) Earthformer: Exploring Space-Time Transformers for Earth System Forecasting[PDF](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a2affd71d15e8fedffe18d0219f4837a-Abstract-Conference.html) | NeurIPS 2022 | MovingMNIST,N -body MNIST  | [Code](https://proceedings.neurips.cc/paper_files/paper/2022/file/a2affd71d15e8fedffe18d0219f4837a-Supplemental-Conference.zip) |
| (Rainformer)Rainformer: Features Extraction Balanced Network for Radar-Based Precipitation Nowcasting[PDF](https://ieeexplore.ieee.org/document/9743916) | IEEE Geoscience and Remote Sensing Letters 2022 | KNMI | [Code](https://github.com/Zjut-MultimediaPlus/Rainformer?tab=readme-ov-file) |
| (MetNet) MetNet: A Neural Weather Model for Precipitation Forecasting [PDF](https://arxiv.org/abs/2003.12140) | arXiv 2020 | - | - |
| (ConvLSTM) Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting[PDF]() | NIPS 2015 | Moving MNIST | [Code](https://proceedings.neurips.cc/paper_files/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Supplemental.zip) |
| (PhysDL) Deep learning for physical processes: incorporating prior scientific knowledge[PDF](https://arxiv.org/abs/1711.07970) | IOPScience 2019 | NEMO | - |
| (PhyDNet) Disentangling Physical Dynamics From Unknown Factors for Unsupervised Video Prediction[PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Le_Guen_Disentangling_Physical_Dynamics_From_Unknown_Factors_for_Unsupervised_Video_Prediction_CVPR_2020_paper.html) | CVPR 2020 | MovingMNIST、TrafficBJ、Sea Surface Temperature、Human3.6 | [Code](https://github.com/vincent-leguen/PhyDNet) |
| GANRain: Skillful precipitation nowcasting using deep generative models of radar[PDF](https://www.nature.com/articles/s41586-021-03854-z) | Nature 2021 | - | [Code](https://github.com/openclimatefix/skillful_nowcasting) |
| MultiScaleGAN: Experimental Study on Generative Adversarial Network for Precipitation Nowcasting[PDF](https://ieeexplore.ieee.org/abstract/document/9780397) | IEEE TRANSACTIONS ON GEOSCIENCE AND REMOTE SENSING 2022 | - | [Code](https://github.com/luochuyao/MultiScaleGAN) |



