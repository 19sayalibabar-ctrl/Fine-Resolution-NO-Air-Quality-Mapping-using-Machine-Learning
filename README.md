# 🌍 Fine-Resolution NO₂ Air Quality Mapping using Machine Learning

> **Downscaling coarse satellite NO₂ data to 1km resolution using multi-source geospatial ML**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Overview

Air pollution is a global health crisis — yet most satellite-based NO₂ measurements are available only at **coarse 7km resolution**, making it impossible to identify street-level hotspots in dense urban environments.

This project develops a **machine learning-based spatial downscaling framework** that takes Sentinel-5P/TROPOMI satellite data as input and generates **fine-resolution (~1km) NO₂ air quality maps** for Indian cities. The model fuses satellite retrievals with ground station data, meteorological variables, land-use data, and road network density to produce high-accuracy, interpretable pollution maps.

**Mentor:** Dr. Ganesh R. Pathak | MIT-ADT University

---

## 🔬 Key Features

- ✅ **Spatial Downscaling** — 7km → 1km NO₂ resolution via ensemble regression
- ✅ **Multi-Source Data Fusion** — Satellite + ground stations + meteorology + land-use + roads
- ✅ **Cloud Gap Imputation** — MICE-based spatiotemporal imputation for cloud-covered pixels
- ✅ **Interactive Dashboard** — Streamlit app with Google Maps tile overlay + live CPCB data
- ✅ **Explainability** — SHAP analysis to identify top pollution drivers
- ✅ **API Integration** — Copernicus Open Access Hub + CPCB live data APIs

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      DATA INGESTION LAYER                        │
│                                                                  │
│  ┌─────────────┐  ┌───────────────┐  ┌──────────┐  ┌────────┐  │
│  │ Sentinel-5P │  │ CPCB Ground   │  │ ERA5 Met │  │ MODIS  │  │
│  │ TROPOMI NO₂ │  │ Station Data  │  │ Variables│  │ LULC   │  │
│  │ (~7km coarse│  │ (hourly µg/m³)│  │ (T, WS,  │  │ (Land  │  │
│  │  resolution)│  │               │  │  BLH, RH)│  │  use)  │  │
│  └──────┬──────┘  └───────┬───────┘  └────┬─────┘  └───┬────┘  │
│         └──────────────────┴───────────────┴────────────┘       │
│                                │                                 │
│                    ┌───────────▼────────────┐                   │
│                    │  Feature Engineering   │                   │
│                    │  & Preprocessing Layer │                   │
│                    └───────────┬────────────┘                   │
│                                │                                 │
│                    ┌───────────▼────────────┐                   │
│                    │   ML Model (Ensemble)  │                   │
│                    │   XGBoost + LightGBM   │                   │
│                    │   + Stacked Regressor  │                   │
│                    └───────────┬────────────┘                   │
│                                │                                 │
│              ┌─────────────────▼──────────────────┐            │
│              │  1km Resolution NO₂ Prediction Map  │            │
│              │  + Streamlit Dashboard + SHAP Maps  │            │
│              └────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
no2-air-quality-mapping/
├── data/
│   ├── satellite/              # Sentinel-5P L2 NO₂ NetCDF files
│   ├── ground_stations/        # CPCB hourly NO₂ CSVs
│   ├── meteorological/         # ERA5 reanalysis data
│   ├── auxiliary/              # MODIS LULC, OSM road density, population
│   └── processed/              # Merged, cleaned feature matrices
├── preprocessing/
│   ├── satellite_reader.py     # Sentinel-5P NetCDF parser + regridding
│   ├── ground_merge.py         # Spatial join: stations to grid cells
│   ├── met_features.py         # ERA5 meteorological feature extraction
│   └── imputation.py           # MICE spatiotemporal gap filling
├── features/
│   ├── spatial_features.py     # Kriging, Moran's I, spatial autocorrelation
│   ├── temporal_features.py    # Lag features, rolling averages
│   └── feature_pipeline.py    # Master feature engineering pipeline
├── models/
│   ├── train.py                # Model training + cross-validation
│   ├── ensemble.py             # Stacked ensemble regressor
│   └── optuna_tuning.py        # Hyperparameter optimisation
├── evaluation/
│   ├── metrics.py              # R², RMSE, MAE, MAPE
│   └── shap_analysis.py        # SHAP feature importance
├── app/
│   ├── streamlit_app.py        # Interactive Streamlit dashboard
│   └── cpcb_api.py             # CPCB live data API connector
├── notebooks/
│   ├── 01_EDA_Satellite.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Comparison.ipynb
│   ├── 04_SHAP_Analysis.ipynb
│   └── 05_Dashboard_Demo.ipynb
├── requirements.txt
└── README.md
```

---

## 🧠 Technical Details

### Data Sources & Integration
| Source | Variable | Resolution | Format |
|---|---|---|---|
| Sentinel-5P / TROPOMI | NO₂ column density | ~7km | NetCDF-4 |
| CPCB Ground Stations | Hourly NO₂ (µg/m³) | Point data | CSV / API |
| ERA5 (ECMWF) | Temperature, Wind Speed, BLH, RH | 31km | GRIB2 |
| MODIS MCD12Q1 | Land-Use / Land-Cover | 500m | HDF4 |
| OpenStreetMap | Road network density | Vector | GeoJSON |
| GPWv4 | Population density | 1km | GeoTIFF |

### Feature Engineering Pipeline
- **Spatial:** Kriging interpolation of ground station NO₂, Moran's I spatial autocorrelation index, distance-to-nearest-highway, urban fraction per grid cell
- **Temporal:** 3h / 6h / 24h rolling mean NO₂, day-of-week encoding, hour-of-day encoding, seasonal indicator
- **Meteorological:** Wind U/V components, planetary boundary layer height (BLH), temperature inversion index, relative humidity, solar radiation
- **Interactions:** BLH × Temperature inversion, road density × wind speed, population density × NO₂ baseline

### Model Comparison Results
| Model | R² | RMSE (µg/m³) | MAE | MAPE |
|---|---|---|---|---|
| **XGBoost (Best)** | **0.89** | **3.2** | **2.4** | **11.2%** |
| LightGBM | 0.87 | 3.5 | 2.6 | 12.4% |
| Stacked Ensemble | 0.88 | 3.3 | 2.5 | 11.8% |
| Random Forest | 0.85 | 3.9 | 2.9 | 13.7% |
| Bilinear Interpolation (Baseline) | 0.61 | 6.1 | 4.8 | 22.3% |

### Top SHAP Feature Contributions
1. Road network density — 28% variance explained
2. Planetary boundary layer height — 22%
3. Temperature inversion index — 20%
4. Population density — 12%
5. Satellite NO₂ (coarse) — 10%
6. Wind speed — 8%

---

## ⚙️ Installation & Setup

```bash
git clone https://github.com/sayali-babar19/no2-air-quality-mapping.git
cd no2-air-quality-mapping
pip install -r requirements.txt
```

### Data Download
```bash
# Download Sentinel-5P data via Copernicus API
python data/satellite/download_tropomi.py --start 2024-01-01 --end 2024-12-31 --region india

# CPCB data via API
python app/cpcb_api.py --cities "Mumbai,Pune,Delhi,Bangalore"
```

### Run Training Pipeline
```bash
python models/train.py --config configs/train_config.yaml --model xgboost
```

### Launch Streamlit App
```bash
streamlit run app/streamlit_app.py
# Open http://localhost:8501
```

---

## 📊 Dashboard Features

The Streamlit dashboard provides:
- 🗺️ **Interactive choropleth map** — 1km NO₂ predictions overlaid on Google Maps tiles
- 📈 **Time-series panel** — Hourly/daily NO₂ trends per city
- 🎚️ **Scenario slider** — Adjust meteorological inputs to simulate pollution scenarios
- 📍 **Station comparison** — Predicted vs. ground-truth CPCB values
- 🔴 **Hotspot alerts** — Flags grid cells exceeding WHO NO₂ thresholds (40 µg/m³ annual mean)

---

## 🔭 Future Work

- [ ] Extend to PM2.5 and O₃ species
- [ ] Integrate with Google Earth Engine for large-scale processing
- [ ] Deploy as a public REST API with city-level endpoints
- [ ] Add health impact modelling (DALY estimates from NO₂ exposure)

---

## 📚 References

- Sentinel-5P / TROPOMI NO₂ Product Documentation — ESA/Copernicus
- ERA5 Reanalysis Data — ECMWF / Copernicus Climate Data Store
- CPCB National Air Quality Index — https://cpcb.nic.in
- Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD.

---

## 👤 Author

**Sayali Babar**  
[https://www.linkedin.com/in/sayali-babar19/](https://www.linkedin.com/in/sayali-babar19/)  
Machine Learning Expert | AI & Neural Networks Practitioner | Data Analytics Practitioner

---

## 📄 License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute this project for educational and research purposes.

```
MIT License — Copyright (c) 2025 Sayali Girish Babar
```
