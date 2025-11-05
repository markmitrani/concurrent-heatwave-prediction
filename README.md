# Predicting weather-driven crop failures with space-time transformers
The github repository for the data pipeline used to predict stream function archetypes related to concurrent heatwaves using Earthformer.

## Installation
Set up virtual environment and install all dependencies automatically:

```./create_venv.sh```

## Usage
This is a multi-step pipeline, so follow it in order for the best results and least headache 🧘‍♂️⚙️✨
### Pre-processing
Takes LENTIC files stored as netCDF files and runs the necessary preprocessing steps on them.

**Preprocessing stream function**

```jobs/run_preprocessing_stream.sh```

**Preprocessing TAS**

```jobs/run_preprocessing_tas.sh```

**Preprocessing OLR**

```jobs/run_preprocessing_olr.sh```
### Archetypal Analysis
**Perform SVD**

To obtain stream function archetypes, an initial dimensionality reduction step using SVD is necessary. The recommended number of dimensions is ```k = 40```.

```jobs/run_SVD.sh```


**Run Archetypal Analysis**

Using the outputs from the SVD step, the archetypes can now be identified:

```jobs/run_AA.sh```

**Visualize AA results**

With the results at hand, we can now visualize them by restoring the original dimensionality with a quick projection step.

```python scripts/view_results_proj.py```

*Note:* 'proj' in the script name refers to placing the results on a Plate Carree projection with continental borders, rather than the dimensionality operations.

### Composite Analysis
**Perform Composite Analysis**

The final step before prediction is to determine the TAS composites associated with each archetype. The resulting plots assign z-scores to archetypes based on temperature anomalies observed within the defined regions of interest (by default set to crop regions within USA and Europe) when that archetype is prevalent.

```python scripts/composite_analysis.py --method weighted```

Adjust method according to desired composite methodology: "argmax" performs hard grouping, "weighted" performs soft composites.

**Interpreting z-scores**

Each of the z-scoring methods are relevant for different anomalous temperature configurations:

- **|z|:** Absolute anomalies (both positive and negative anomalies are considered equally).

- **z+:** Positive anomalies (warmer temperatures than average)

- **z-:** Negative anomalies (cooler temperatures than average)

### Prediction
Once the most relevant archetype for prediction is chosen based on the composites, the prediction procedure takes an input sequence containing stream function and OLR data for ```input_len``` days and returns a score between 0 and 1 corresponding to the foreseen prevalence of the chosen archetype with an lead of ```lead_time``` days into the future. The pipeline uses an Earthformer model as base, and appends a lightweight predictor head to transform the model output into a probability score. The steps of the pipeline are broken down into various files for modularity and a clear organization:

```prediction/data.py```: Merging stream function and OLR data, preparing x and y tensors and DataLoaders for model use.

```prediction/model.py```: Model definitions with Earthformer as a base model and EarthformerPredictor as the final model used in the pipeline.

```prediction/run.py```: Bringing everything together, setting (hyper)parameters, calling data and model helpers, running the training procedure.

```prediction/utils.py```: Various utilities to plot relevant information during training.

**Run Prediction Pipeline**

```python prediction/run.py```

**Parameter Settings**

Adjust the following parameters as necessary to test various experimental setups:

```input_len```: Number of past timesteps provided as input to the model. Controls how much historical context the network uses to make predictions.

```lead_time```: Forecast horizon (in days). Determines how far ahead the model predicts relative to the last input timestep.

```batch_size```: Number of samples processed simultaneously during training. Larger batches improve training stability but require more GPU memory.

```num_epochs```: Total number of full training passes over the dataset. Increasing this can improve convergence but risks overfitting.

```archetype_index```: Specifies which archetype is targeted for prediction.

```olr_lag```: Lag (in days) applied to Outgoing Longwave Radiation (OLR) features to capture delayed atmospheric responses or teleconnections.

```rolling_avg_window```: Window length (in days) for computing rolling averages of the target variable. Smooths short-term variability and highlights persistent signals.

*NOTE:* Make sure the input and output directories and variable names point to the correct paths in the scripts. The relevant variables should be configured under ```/config.env``` to consistently sync across different jobs. If parts of the procedure are changed, some python scripts may still require a minimal amount of manual configuration for differing file names.
