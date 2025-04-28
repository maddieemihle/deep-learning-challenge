# Deep Learning Challenge

## Introduction 
Alphabet Soup, a nonprofit foundation, aims to identify applicants for funding who are most likely to succeed in their ventures. Using historical data of over 34,000 organizations, this project builds a binary classification model to predict whether an applicant will be successful if funded. The goal is to assist Alphabet Soup in making data-driven funding decisions.

This project uses a deep learning neural network to analyze the dataset and attempts to optimize the model to achieve an accuracy of 75% or higher.

## Challenge Instructions: 
**Step 1: Import Dependencies and Load Dataset**
* Include all necessary libraries (e.g., `pandas`, `tensorflow`, `sklearn`) and load the dataset into a DataFrame.

**Step 2: Inspect and Preprocess the Data**
* Drop Non-Beneficial Columns: Remove columns like `EIN`, `NAME`, and others that do not contribute to the prediction.
* Analyze Unique Values in Categorical Columns: Identify columns with many unique values to determine if rare categories need to be grouped.
* Combine Rare Categories: Group rare categories into an "Other" category for columns like `APPLICATION_TYPE` and `CLASSIFICATION`.
* Encode Categorical Variables: Use one-hot encoding (`pd.get_dummies`) to convert categorical variables into numerical format.
* Split Data into Features and Target: Separate the target variable (`IS_SUCCESSFUL`) from the features.
* Scale Features: Use `StandardScaler` to normalize numerical features for better model performance.

**Step 3: Build and Train the Neural Network**
* Define the Model Architecture: Create the initial neural network with input, hidden, and output layers.
* Compile and Train the Model: Compile the model with an optimizer (e.g., Adam), loss function (e.g., binary crossentropy), and evaluation metric (e.g., accuracy). Train the model on the training dataset.
* Save Model Weights with Callbacks: Use callbacks like `ModelCheckpoint` to save the model weights during training.
* Evaluate Model Performance: Evaluate the model on the test dataset to calculate accuracy and loss.

**Step 4: Optimize the Model**
* Create a new notebook to train and experiment models to optimize 
* Adjust Input Data: Experiment with dropping additional columns, combining categories, or applying transformations.
* Modify Neural Network Architecture: Add more layers, neurons, Batch Normalization, or Dropout to improve performance.
* Retrain and Evaluate the Optimized Model

**Step 5: Save Final Model**
* Save the final trained model to an HDF5 file

**Step 6: Write A Report**
* Summarize the results, challenges, and recommendations for future improvements.


## Report on the Neural Network Model and Results

### **Overview of the Analysis**
The purpose of this analysis is to create a binary classification model using a deep learning neural network to predict whether applicants for funding from Alphabet Soup will be successful in their ventures. By analyzing historical data, the model aims to assist Alphabet Soup in making data-driven decisions to allocate funding to applicants with the highest likelihood of success.

### **Results**

##### Part 1: Data Processing
**Question 1: What variable(s) are the target(s) for your model?**
* The target variable is `IS_SUCCESSFUL` which indicates whethere the funding was used effectively for success or for failure 

**Question 2: What variable(s) are the features for your model?**
The variables that the features include are: 
* `APPLICATION_TYPE`
* `AFFILIATION`
* `CLASSIFICATION`
* `USE_CASE`
* `ORGANIZATION`
* `INCOME_AMT`
* `ASK_AMT` (eventually droped this variable, but it did not have that much of an effect in the end)

**Question 3: What variable(s) should be removed from the input data because they are neither targets nor features?**
* `EIN` and `NAME`: Identification columns that do not contribute to the prediction.
* `STATUS`: Had little variance and did not provide meaningful information.
* `SPECIAL_CONSIDERATIONS`: Highly imbalanced and did not significantly impact the model.
* `ASK_AMT`: Transformed using a log transformation to reduce the impact of outliers.

##### Part 2: Compiling, Training, and Evaluating the Model

Overall, when training and comiling the data and models, four different attempts were made to try and get the accuracy over 75%. The questions are answered as follows: 

**Question 1: How many neurons, layers, and activation functions did you select for your neural network model, and why?**

**1st Optimization Model**
* Hidden Layers:
    - Layer 1: 128 neurons, ReLU activation, and Dropout (20%).
    - Layer 2: 64 neurons, ReLU activation, and Dropout (20%).
* Output Layer: 1 neuron with a sigmoid activation function for binary classification.
* Reasoning:
    - A simple architecture was chosen to establish a baseline for performance.
    - ReLU activation was used to handle non-linear relationships.
    - Dropout was added to prevent overfitting.

Image: 1st Opt Model 
![1st attempt](https://github.com/maddieemihle/deep-learning-challenge/blob/main/Images/Opt_1.png?raw=true)

--------
**2nd Optimization Model**
* Hidden Layers:
    - Layer 1: 256 neurons, ReLU activation, and Dropout (30%).
    - Layer 2: 128 neurons, ReLU activation, and Dropout (30%).
    - Layer 3: 64 neurons, ReLU activation, and Dropout (20%).
* Output Layer: 1 neuron with a sigmoid activation function for binary classification.
* Reasoning:
    - Increased the number of neurons and layers to allow the model to learn more complex patterns.
    - Dropout rates were increased to prevent overfitting.

Image: 2nd Opt Model
![2nd attempt](https://github.com/maddieemihle/deep-learning-challenge/blob/main/Images/Opt_2.png?raw=true)

--------
**3rd Optimization Model**
* Hidden Layers:
    - Layer 1: 256 neurons, ReLU activation, Batch Normalization, and Dropout (30%).
    - Layer 2: 128 neurons, ReLU activation, Batch Normalization, and Dropout (30%).
    - Layer 3: 64 neurons, ReLU activation, Batch Normalization, and Dropout (20%).
    - Layer 4: 32 neurons, ReLU activation, Batch Normalization, and Dropout (20%).
* Output Layer: 1 neuron with a sigmoid activation function for binary classification.
* Reasoning:
    - Added Batch Normalization to stabilize and speed up training.
    - Increased the number of layers to capture more complex relationships.
    - Dropped additional columns (`ASK_AMT`) to simplify the input data.

Image: 3rd Opt Model 
![3rd attempt](https://github.com/maddieemihle/deep-learning-challenge/blob/main/Images/Opt_3.png?raw=true)

--------
**4th Optimization Model**
* Hidden Layers:
    - Layer 1: 256 neurons, ReLU activation, Batch Normalization, and Dropout (30%).
    - Layer 2: 128 neurons, ReLU activation, Batch Normalization, and Dropout (30%).
    - Layer 3: 64 neurons, ReLU activation, Batch Normalization, and Dropout (20%).
    - Layer 4: 32 neurons, ReLU activation, Batch Normalization, and Dropout (20%).
* Output Layer: 1 neuron with a sigmoid activation function for binary classification.
* Reasoning: 
    - ReLU activation was chosen for hidden layers to handle non-linear relationships.
    - Batch Normalization was added to stabilize and speed up training.
    - Dropout was used to prevent overfitting.
    - The architecture was tuned using Keras Tuner to find the optimal number of neurons and layers.

Image: 4th Opt Model 
![4th attempt](https://github.com/maddieemihle/deep-learning-challenge/blob/main/Images/opt_4.png?raw=true)

--------

**Question 2: Were you able to achieve the target model performance?**
* Despite multiple optimizations, the model achieved a maximum accuracy of 73%, which is below the target of 75%. The model performace is as follows: 
    * 1st Optimization Attempt: 
        - Accuracy 73%
        - Loss 56% 
        - Initial model with basic architecture and preprocessing. This attempt achieved the highest accuracy among all attempts.
    * 2nd Optimization Attempt: 
        - Accuracy 72% 
        - Loss 56% 
        - Adjusted the model architecture by adding more layers and neurons, but the accuracy slightly decreased, likely due to overfitting.
    * 3rd Optimization Attempt:     
        - Accuracy 72%
        - Loss 55%
        - Dropped additional columns and added more parameters to the model. However, this did not improve accuracy, indicating that the additional parameters may not have been meaningful.
    * 4th Optimization Attempt: 
        - Accuracy 72%
        - Loss 55% 
        - Used Hyperparameter Tuning to optimize the number of neurons, dropout rates, and learning rate. Despite systematic tuning, the accuracy remained the same as the third attempt.

**Question 3: What steps did you take in your attempts to increase model performance?**
* Various steps were taken and adjustments were made to attempt to increase models performance. They are listed as follows: 
    * Data Preprocessing:
        - Combined rare categories in `APPLICATION_TYPE` and `CLASSIFICATION` into an "Other" category.
        - Dropped additional columns (`STATUS`, `SPECIAL_CONSIDERATIONS`) to remove irrelevant or low-variance features.
    * Model Architecture:
        - Increased the number of neurons and layers.
        - Added *Batch Normalization* and *Dropout* to improve generalization and prevent overfitting.
    * Hyperparameter Tuning:
        - Used *Keras Tuner* to optimize the number of neurons, dropout rates, and learning rate.
        - Tried different activation functions (ReLU, LeakyReLU, tanh).
    * Learning Rate Adjustment:
        - Used a learning rate scheduler (`ReduceLROnPlateau`) to dynamically adjust the learning rate during training.
    * Experimentation with Columns:
        - Dropped additional columns and tested their impact on accuracy (`ASK_AMT`)


### Summary
Overall Results: The deep learning model achieved a maximum accuracy of 73%, which is below the target of 75%. Despite extensive optimizations, the model struggled to extract enough meaningful patterns from the data to achieve the desired performance.

| Attempt | Hidden Layers | Accuracy | Loss | Key Changes |
|---------|---------------|----------|------|-------------|
| 1st     | 2             | 73%      | 56%  | Baseline model with simple architecture. |
| 2nd     | 3             | 72%      | 56%  | Added more layers and neurons, but overfitting occurred. |
| 3rd     | 4             | 72%      | 55%  | Added Batch Normalization and dropped columns. |
| 4th     | 4             | 72%      | 55%  | Used Hyperparameter Tuning to optimize architecture. |

### Recommendation
* When attempting the Batch Normalization and Dropout, I hoped that I would see a difference as in the 10 trials there was an increse to 74%. However, when fitting the model after the trials were made, it jumped back down to 72%. I think further analysis could help improve this. 
* A different machine learning model, such as Random Forest or Gradient Boosting (e.g., XGBoost), may be more suitable for this classification problem. These models often perform better on tabular data with categorical features and can handle imbalanced datasets effectively.
* Additionally, feature engineering (e.g., creating interaction terms, binning numerical variables) and further exploration of the dataset (e.g., identifying additional patterns or correlations) could improve performance.

### Conclusion
Overall, neural networks excel at learning from large datasets with complex, non-linear relationships (e.g., image or text data). However, for structured tabular data, tree-based models like Random Forest or Gradient Boosting often outperform neural networks due to their ability to handle categorical variables and imbalanced data more effectively.

## Tools & Technologies used 
* **Development Environment**: Jupyter Notebook / Google Colab
* **Version Control**: Git and GitHub
* **Hardware**: Local machine or cloud-based GPU for faster training.

## Resources
- **Dataset**: Provided by Alphabet Soup (CSV file containing metadata about organizations).
- **Documentation**:
  - [TensorFlow Documentation](https://www.tensorflow.org/)
  - [Keras Tuner Documentation](https://keras.io/keras_tuner/)
  - [Pandas Documentation](https://pandas.pydata.org/)
  - [Scikit-learn Documentation](https://scikit-learn.org/)
- **References**:
  - IRS. Tax Exempt Organization Search Bulk Data Downloads. [External Link](https://www.irs.gov/charities-non-profits/exempt-organizations-business-master-file-extract-eo-bmf)
