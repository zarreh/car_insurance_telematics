# Telematics Data Processing Pipeline

## Overview
Production-ready pipeline for processing raw telematics sensor data from mobile devices into driver risk features.

## Project Structure
```
telematics-pipeline/
├── config.py              # Configuration management
├── preprocess.py          # Core preprocessing logic
├── data_loader.py         # Data I/O utilities
├── feature_engineering.py # Driver-level aggregation
├── main.py               # Entry point
├── utils.py              # Helper functions
├── data/
│   ├── raw/             # Input JSON files
│   └── processed/       # Output CSV files
└── logs/                # Processing logs
```

## Usage

### Basic Processing
```bash
python main.py --input-dir data/raw --output-dir data/processed
```

### Full Pipeline with Driver Aggregation
```bash
python main.py \
    --input-dir data/raw \
    --output-dir data/processed \
    --aggregate-drivers \
    --batch-size 100 \
    --log-level INFO
```

### Command Line Options
- `--input-dir`: Directory containing raw JSON sensor files
- `--output-dir`: Directory for processed CSV outputs
- `--batch-size`: Number of trips to process at once (default: 100)
- `--save-intermediate`: Save results after each batch
- `--aggregate-drivers`: Create driver-level features
- `--log-level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

## Data Flow

1. **Raw JSON Files** → Loaded by `DataLoader`
2. **Sensor Data** → Processed by `TripProcessor`
3. **Trip Summaries** → Saved as CSV
4. **Driver Features** → Aggregated by `DriverAggregator`

## Output Files

- `all_processed_trips.csv`: Trip-level features
- `driver_features.csv`: Driver-level aggregated features
