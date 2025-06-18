# Dynamic community detection algorithm in fMRI brain graphs with node features

The goal is to detect evolving functional communities in the brain over time by:
- Building dynamic functional brain networks from fMRI time series
- Incorporating anatomical and signal-based node features
- Developing or adapting community detection algorithms to handle time-evolving and feature-rich graph data

```bash
├── data/                   # (not tracked)
│   ├── raw/                
│   └── processed/          
│
├── notebooks/              
│   ├── preprocessing.ipynb # Building dynamic graphs from raw time series
│   └── features_extraction.ipynb # Computing node features 
│
├── src/                    # Source code modules
│   ├── preprocessing.py    # Raw timeseries preprocessing functions
│   ├── graph_construction.py # Graph creation and correlation matrix handling
│   ├── features.py         # Node feature extraction functions
│   ├── utils.py            # Helpers 
│
├── results/               
│
├── README.md               
└── requirements.txt        
