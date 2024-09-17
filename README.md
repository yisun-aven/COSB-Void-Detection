
# COSB Void Detection

This project focuses on detecting voids in Composite Oriented Strand Boards (COSBs) using advanced image segmentation and classification techniques. It leverages semi-supervised learning models such as k-means clustering and XGBoost classifiers to enhance the detection process.

## Project Overview

This notebook explores a workflow for detecting voids in COSBs using a combination of:
- Data preprocessing
- Machine learning models
- Evaluation metrics

### Key Techniques and Libraries:
- **KMeans Clustering**: For clustering of void regions.
- **XGBoost Classifier**: For supervised learning and classification.
- **Image Processing**: Extracting features and preparing input for machine learning models.
- **Performance Metrics**: Evaluation using metrics like accuracy, precision, recall, etc.

## Project Setup

1. **Google Drive Integration**: The project leverages Google Drive for data storage.
   - Make sure you have the required data files in the correct path within your Drive.
   - Modify the paths accordingly in the notebook if your directory structure is different.
   
2. **Dependencies**:
   - The notebook requires several libraries such as `xgboost`, `sklearn`, `numpy`, `matplotlib`, etc.
   - To install dependencies, use the following command:
     ```bash
     !pip install -r requirements.txt
     ```

3. **Data Preprocessing**:
   - StandardScaler is used for feature scaling.
   - Train-test split is performed to divide the data for training and evaluation.

4. **Model Training**:
   - The models are trained on preprocessed features from COSB images.
   - Hyperparameters are tuned using `GridSearchCV`.
   - Several models were tested to compare their effectiveness. Ultimately, the semi-supervised approach using k-means clustering and XGBoost performed the best.

5. **Evaluation**:
   - Classification metrics such as precision, recall, accuracy, and Jaccard score are used to evaluate model performance.

## Running the Notebook

1. **Mount Google Drive**: Run the following command to mount your Google Drive and access the necessary files.
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Data Processing and Feature Extraction**:
   - The `image_processing.py` script is used to extract features from the COSB images.
   - Additional scripts, such as `run_knn.py`, are used to fine-tune the models.

3. **Model Training and Evaluation**:
   - Train the models and evaluate their performance on the labeled data using various classification metrics.

## Results

The project outputs confusion matrices, precision-recall curves, and other performance metrics to evaluate the effectiveness of the models in detecting voids in COSB.

After experimenting with several different machine learning models, the semi-supervised approach using k-means clustering for feature grouping and XGBoost for classification yielded the best results in terms of accuracy and precision.
