# BERTrend Topic Analysis Demo

## Overview

The `bertrend.demos.topic_analysis` package provides a Streamlit-based web application for topic analysis using BERTopic. This application allows users to load data, train topic models, explore topics, visualize topic distributions and relationships, analyze temporal trends, and generate newsletters based on the analysis.

## Features

The Topic Analysis Demo application offers the following key features:

1. **Data Loading & Model Training**
   - Load data from various sources
   - Train BERTopic models on the loaded data
   - Save and load trained models

2. **Topic Exploration**
   - Browse and search topics
   - View documents associated with specific topics
   - Analyze topic content and keywords

3. **Topic Visualization**
   - Visualize topic hierarchies
   - Display topic distributions
   - Explore topic relationships

4. **Temporal Visualization**
   - Analyze topic evolution over time
   - Track topic popularity trends
   - Identify emerging and declining topics

5. **Newsletter Generation**
   - Generate newsletters based on topic analysis
   - Customize newsletter content and format

## Components

The package consists of several key components:

### Main Application

The main application (`app.py`) provides a Streamlit-based web interface with multiple pages:
- "Data loading & model training" - for loading data and training models
- "Topic exploration" - for exploring topics and associated documents
- "Topic visualization" - for visualizing topic relationships and distributions
- "Temporal visualization" - for analyzing topic evolution over time
- "Newsletter generation" - for creating newsletters based on the analysis

### Utility Functions

The package includes utility functions (`app_utils.py`) for:
- Computing topics over time
- Displaying documents for specific topics
- Splitting data for temporal analysis

### Demo Pages

The package includes several demo pages in the `demo_pages` directory:

- `training_page.py` - For loading data and training models
- `explore_topics.py` - For exploring topics and associated documents
- `topic_visualizations.py` - For visualizing topic relationships and distributions
- `temporal_visualizations.py` - For analyzing topic evolution over time
- `newsletters_generation.py` - For creating newsletters based on the analysis

### Messages

The package includes predefined messages (`messages.py`) for user feedback and error handling.

## Usage

### Starting the Application

To start the Topic Analysis Demo application:

```bash
cd bertrend/demos/topic_analysis
streamlit run app.py
```

### Data Loading & Model Training

Users can load data and train models in the "Data loading & model training" page:
- Upload data files
- Configure model parameters
- Train and save models

### Topic Exploration

The "Topic exploration" page allows users to:
- Browse topics by relevance
- Search for specific topics
- View documents associated with each topic
- Analyze topic content and keywords

### Topic Visualization

The "Topic visualization" page provides:
- Hierarchical visualizations of topics
- Topic distribution charts
- Interactive topic maps

### Temporal Visualization

The "Temporal visualization" page enables:
- Analysis of topic evolution over time
- Tracking of topic popularity trends
- Identification of emerging and declining topics

### Newsletter Generation

The "Newsletter generation" page allows users to:
- Generate newsletters based on topic analysis
- Customize newsletter content and format
- Export newsletters in various formats

## Dependencies

The Topic Analysis Demo package depends on:
- BERTrend core functionality
- BERTopic for topic modeling
- Streamlit for the web interface
- Various data processing and visualization libraries

These dependencies are included in the main package requirements.