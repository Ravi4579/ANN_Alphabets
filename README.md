# Classification Using Artificial Neural Networks with Hyperparameter Tuning

## Overview
This project demonstrates the use of Artificial Neural Networks (ANNs) to classify alphabets using the "Alphabets_data.csv" dataset. The implementation includes data preprocessing, model development, hyperparameter tuning, and performance evaluation.

## Dataset
The dataset "Alphabets_data.csv" contains labeled data for a classification task to identify different alphabets. The dataset underwent preprocessing to ensure optimal performance, including normalization and handling of missing values.

## Project Structure
```
.
├── Alphabets_data.csv        # Dataset file
├── main.py                   # Main script for loading data, training the model, and evaluation
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
└── results
    ├── model_summary.txt     # Summary of the ANN model architecture
    └── evaluation_metrics.txt # Metrics for the model's performance
```

## Requirements
- Python 3.8 or later
- TensorFlow
- Keras
- NumPy
- pandas
- scikit-learn
- matplotlib

Install dependencies using:
```
pip install -r requirements.txt
```

## Steps to Run the Project
1. **Clone the repository:**
   ```
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Place the dataset:**
   Ensure that the `Alphabets_data.csv` file is in the project root directory.

3. **Run the main script:**
   ```
   python main.py
   ```

   The script will load the dataset, preprocess the data, train the ANN model, tune hyperparameters, and evaluate the model.

4. **View results:**
   - The model architecture summary will be saved in `results/model_summary.txt`.
   - Performance metrics will be saved in `results/evaluation_metrics.txt`.

## Key Features
### 1. Data Exploration and Preprocessing
- **Exploration:** Summarized the dataset, including the number of samples, features, and classes.
- **Preprocessing:**
  - Normalized the data to bring features to a similar scale.
  - Handled missing values by appropriate imputation.

### 2. ANN Model Implementation
- Constructed a basic ANN model with at least one hidden layer using TensorFlow and Keras.
- Divided the dataset into training and test sets for model evaluation.
- Trained the ANN model using a suitable optimizer and loss function.

### 3. Hyperparameter Tuning
- Used Grid Search and Random Search techniques to optimize:
  - Number of hidden layers
  - Neurons per hidden layer
  - Activation functions
  - Learning rates
- Documented the tuning process and outcomes.

### 4. Evaluation
- Evaluated model performance using metrics such as accuracy, precision, recall, and F1-score.
- Compared the performance of the default model with the tuned model.

## Results
- The tuned ANN model achieved improved performance metrics compared to the default model.
- Key findings on the impact of hyperparameters on model accuracy and generalization are documented in `results/evaluation_metrics.txt`.

## Discussion
The project highlights the importance of data preprocessing and structured hyperparameter tuning in achieving superior model performance. The tuned ANN model outperformed the default model significantly, demonstrating the effectiveness of the applied optimization techniques.

## References
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---


