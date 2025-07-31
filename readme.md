# Predicting Weather-Driven Crop Failures with Space-Time Transformers

Welcome to the GitHub repository for the data pipeline designed to predict stream function archetypes related to concurrent heatwaves using the Earthformer model. This project aims to analyze and predict weather-driven crop failures by leveraging advanced machine learning techniques.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains scripts and tools to perform predictive analysis using archetypal analysis (AA) and singular value decomposition (SVD) on meteorological data, specifically related to concurrent heatwaves. The model utilizes spatial and temporal data transformations to yield meaningful insights for agricultural planning and risk mitigation.

## Directory Structure

Here’s a brief overview of the directory structure:

├── check_git.sh                   # Script to check for large files in git
├── config.env                     # Configuration environment variables
├── create_venv.sh                 # Script to create a virtual environment
├── data                           # Folder containing datasets and preprocessing information
│   ├── deseason
│   ├── deseason_smsub
│   ├── first_run
│   └── lat30-60
├── jobs                           # Job scripts for batch processing
├── plots                          # Generated plots from analysis
├── results                        # Result outputs from model training
├── requirements_conda.txt         # Conda dependencies
├── requirements_pip.txt           # Pip dependencies
└── scripts                        # Python scripts for analysis, preprocessing, and predictions

## Using Pip
pip install -r requirements_pip.txt

## Usage
Setup the Environment: Use the script to create and activate a virtual environment.
Preprocessing Data:
Use the preprocessing scripts available in the folder to prepare your datasets before model training.
Run the Model:
Execute the script to run the entire analysis pipeline, starting with preprocessing data, followed by SVD, and archetypal analysis.
Visualize the Results: Generated plots and analysis results will be saved in the and directories.

## Contributing
Contributions to improve the project or add more functionalities are welcome! Please create an issue or [submit a pull request](https://github.com/markmitrani/concurrent-heatwave-prediction/pulls) with a brief description of your changes.

## License
This project is licensed under the [MIT License](https://en.wikipedia.org/wiki/MIT_License). See the  file for more details.
