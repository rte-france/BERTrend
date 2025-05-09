# BERTrend Weak Signals Demo

## Overview

The `bertrend.demos.weak_signals` package provides a comprehensive web application for detecting and analyzing weak signals in topic models over time using BERTrend's topic modeling capabilities. This application allows users to load data, train models, and analyze the evolution of topics and signals over time.

## Features

The Weak Signals Demo application offers the following key features:

1. **Data Loading and Embedding**
   - Load and preprocess textual data
   - Embed documents using configurable embedding models

2. **Model Training**
   - Train BERTopic models for specific time periods
   - Merge models to track topic evolution over time

3. **Signal Analysis**
   - Identify and categorize signals as noise, weak signals, or strong signals
   - Analyze signal evolution over time
   - Perform detailed analysis of individual signals

4. **Topic Evolution Visualization**
   - Visualize topic evolution using Sankey diagrams
   - Track newly emerged topics
   - Monitor topic popularity evolution

5. **State Management**
   - Save and restore application state
   - Cache management for improved performance

## Components

The package consists of several key components:

### Main Application

The main application (`app.py`) provides a Streamlit-based web interface with multiple tabs:
- "Data Loading" - for loading and embedding textual data
- "Model Training" - for training and merging topic models
- "Results Analysis" - for analyzing signals and topic evolution

### Visualization Utilities

The package includes extensive visualization utilities (`visualizations_utils.py`):
- Sankey diagrams for topic evolution
- Signal categorization displays
- Topic popularity evolution charts
- Signal analysis visualizations
- Topic count retrieval and display

### User Messages

The package includes a comprehensive set of user messages (`messages.py`) for:
- Success notifications
- Error warnings
- Progress indicators

## Usage

### Starting the Application

To start the Weak Signals Demo application:

```bash
cd bertrend/demos/weak_signals
streamlit run app.py
```

### Data Loading and Embedding

1. In the "Data Loading" tab:
   - Load your textual data
   - Configure embedding parameters
   - Embed the documents

### Model Training

1. In the "Model Training" tab:
   - Configure BERTopic hyperparameters
   - Train models for specific time periods
   - Merge models to track topic evolution

### Signal Analysis

1. In the "Results Analysis" tab:
   - View signal categorization (noise, weak signals, strong signals)
   - Analyze topic evolution using Sankey diagrams
   - Examine newly emerged topics
   - Track topic popularity evolution
   - Perform detailed analysis of individual signals

### State Management

The application provides state management capabilities:
- Save the current application state
- Restore a previous application state
- Purge cache to free up resources

## Configuration

The application provides configuration options for:
- Embedding hyperparameters
- BERTopic hyperparameters
- BERTrend hyperparameters

These can be configured through the sidebar in the application.

## Dependencies

The Weak Signals Demo package depends on:
- BERTrend core functionality
- BERTopic for topic modeling
- Streamlit for the web interface
- Plotly for interactive visualizations
- Pandas for data manipulation