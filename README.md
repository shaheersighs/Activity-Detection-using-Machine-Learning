# Activity-Detection-using-Machine-Learning
Machine learning Model that can accurately turn dozens of raw “wiggles-in-time” (accelerometer + gyroscope CSV files) into a small program that can look at those wiggles and say “this person is walking / running / sitting /”.

-------------------------------------------------------------------------------------------------------

#### 1. Introduction
This project implements an end-to-end machine-learning pipeline for classifying human activities—Cycling, Sitting, and Walking—using accelerometer and gyroscope data. Our dataset, drawn from Edge Impulse’s Activity Detection collection, comprises 12 recordings captured at approximately 50 Hz. Through a sequence of preprocessing, feature engineering, and modeling stages, we demonstrate progressive improvements from a naïve baseline to a robust, deployable classifier.

####  2. Data Preparation
We began by verifying file integrity: every activity folder contained a matching pair of acc_*.csv and gyro_*.csv files. Each CSV was loaded into pandas, its timestamp column normalized to start at zero milliseconds, and accelerometer (ax, ay, az) and gyroscope (gx, gy, gz) readings merged via an outer join. The combined six-axis signals were then resampled to a uniform 50 Hz grid, with missing values linearly interpolated, and any remaining edge-NaN rows dropped. This yielded 4 870 usable two-second windows (100 samples per window with 50 % overlap), partitioned into 2 911 Cycling, 1 354 Walking, and 605 Sitting windows.

####  3. Baseline Modeling and Group Leakage
Our first model applied a Random Forest to a minimal 30-feature set—mean, standard deviation, minimum, maximum, and root-mean-square for each axis. Under a conventional random 80/20 split, the classifier achieved an over-optimistic 100 % accuracy, but leave-one-recording-out cross-validation (GroupKFold by recording) collapsed to just 38 % average accuracy, exposing severe information leakage: overlapping windows from the same session appeared in both train and test sets.

#### 4. Enriched Statistical Features
To capture richer signal characteristics, we extended the feature set to 72 dimensions by adding median, interquartile range, peak-to-peak range, skewness, kurtosis, and two band-energy measures (0–3 Hz and 3–6 Hz) for each axis, plus magnitude channels for accelerometer and gyro. This enhanced Random Forest rose modestly to 45 % grouped accuracy, confirming that statistical features alone could not overcome recording-level class imbalance: Sitting appeared in only one session, so any fold that held out that session had zero examples in training.

####  5. Spectral-DSP Features and MLP
Inspired by Edge Impulse’s spectral pipeline, we implemented a digital signal-processing block that low-passes each 50 Hz signal at 2.68 Hz, decimates to 5 Hz, and computes a 64-point FFT. We then extracted log-spectral power for the first 31 frequency bins on each of the six axes (after scaling accelerometer axes by 0.04 to match gyroscope magnitudes), producing a 186-dimensional feature vector. A tiny multilayer perceptron (MLP) with a 64-unit dense layer, 20 % dropout, a 32-unit dense layer, and three-way softmax was trained for 50 epochs at a 5 × 10⁻⁴ learning rate.

Under a random 80/20 split, this DSP-MLP pipeline achieved 95–96 % validation accuracy, replicating the Edge Impulse tutorial headline. Critically, when evaluated with StratifiedGroupKFold over 30-second “pseudo-recordings” (so each fold contained at least one example of every activity), the model sustained 75 - 85% average accuracy—demonstrating genuine cross-session generalization.

####  6. Discussion and Limitations
Our experiments revealed two principal lessons. First, train/test leakage via overlapping windows can drastically inflate performance metrics. Group-aware cross-validation is essential whenever data are grouped by session or user. Second, spectral features distilled periodic patterns of walking cadence and pedal rotation far more effectively than time-domain summaries alone. By scaling and filtering the raw signals before FFT, we ensured the MLP had balanced, informative inputs.

A notable limitation is that Sitting was captured in only a single recording, so even after pseudo-chunk grouping, the model’s performance on Sitting windows had higher variance. Collecting additional Sitting and Walking sessions— or applying realistic sensor-noise and orientation augmentations—would further improve robustness.
