# BERTrend Prospective Demo

## Overview

The `bertrend_apps.prospective_demo` package provides a comprehensive web application for prospective analysis using BERTrend's topic modeling and signal detection capabilities. This application allows users to monitor information sources, analyze trends and signals, and generate reports based on the analysis.

## Features

The Prospective Demo application offers the following key features:

1. **Information Source Management**
   - Configure and manage data feeds
   - Monitor data collection status

2. **Model Management**
   - Train new models for specific time periods
   - Monitor model status by information source

3. **Signal Analysis**
   - Identify and analyze weak and strong signals
   - Visualize signal trends over time

4. **Dashboard Analysis**
   - Analyze topic evolution
   - Perform multifactorial analysis

5. **Report Generation**
   - Generate comprehensive reports based on the analysis
   - Customize report content and format

## Components

The package consists of several key components:

### Main Application

The main application (`app.py`) provides a Streamlit-based web interface with multiple tabs:
- "Veilles" (Watches/Monitoring) - for configuring data sources and checking data collection status
- "Modèles" (Models) - for monitoring models by watch
- "Tendances" (Trends) - for signal analysis
- "Analyses" - for dashboard analysis
- "Génération de rapports" (Report Generation) - for creating reports

### Data Processing

The package includes utilities for processing new data (`process_new_data.py`):
- Loading data for specific models and users
- Training new models for specific time periods
- Generating LLM interpretations of the models

### Authentication

The application includes authentication functionality (`authentication.py`) to manage user access and personalize the experience.

### Dashboard Components

Multiple dashboard components provide specialized visualizations and interfaces:
- `dashboard_analysis.py` - For analyzing topic evolution and trends
- `dashboard_signals.py` - For analyzing weak and strong signals
- `dashboard_common.py` - Common utilities for dashboard components

### Feed Management

The package includes utilities for managing information feeds:
- `feeds_config.py` - For configuring information sources
- `feeds_data.py` - For displaying data status
- `feeds_common.py` - Common utilities for feed management

### Report Generation

The package provides functionality for generating reports:
- `report_generation.py` - For creating reports based on the analysis
- `report_utils.py` - Utilities for report generation
- `report_template.html` - HTML template for reports

## Usage

### Starting the Application

To start the Prospective Demo application:

```bash
cd bertrend_apps/prospective_demo
CUDA_VISIBLE_DEVICES=<gpu_number> streamlit run app.py
```

### Authentication

The application includes authentication functionality. Users need to log in with their credentials to access the application. Authentication can be disabled for development purposes by setting `AUTHENTIFICATION = False` in `app.py`.

### Data Configuration

Users can configure information sources in the "Veilles" tab. This includes:
- Adding new information sources
- Configuring data collection parameters
- Monitoring data collection status

### Model Training

New models can be trained using the command-line interface:

```bash
python -m bertrend_apps.prospective_demo.process_new_data train-new-model <user_name> <model_id>
```

Or through the web interface in the "Modèles" tab.

### Analysis

The application provides multiple analysis options:
- Signal analysis in the "Tendances" tab
- Dashboard analysis in the "Analyses" tab

### Report Generation

Users can generate reports based on the analysis in the "Génération de rapports" tab.

## Configuration

The package uses several configuration settings defined in `__init__.py`:
- `CONFIG_FEEDS_BASE_PATH` - Path for user feed configurations
- `BASE_MODELS_DIR` - Path for user models
- `DEFAULT_GRANULARITY` - Default time granularity for analysis
- `DEFAULT_WINDOW_SIZE` - Default window size for analysis

## Dependencies

The Prospective Demo package depends on:
- BERTrend core functionality
- Streamlit for the web interface
- Various data processing and visualization libraries

These dependencies are included in the "apps" optional dependency group in the main package.