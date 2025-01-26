Churn Prediction Using Artificial Neural Networks
Project Overview

This project aims to predict customer churn using an Artificial Neural Network (ANN). The model was trained on a dataset containing customer information, and various visualizations were created to evaluate and interpret the model's performance.

The project involves:

    Data preprocessing, including feature encoding and scaling.
    Building and training a neural network using TensorFlow/Keras.
    Evaluating the model's performance with metrics like accuracy and loss.
    Visualizing results to gain insights into the model and the dataset.

Dataset

The dataset used is Churn_Modelling.csv, which includes:

    Customer demographic and account details.
    A binary target variable indicating churn (1 = Churn, 0 = Not Churn).

Technologies Used

    Python
    Pandas and NumPy for data manipulation.
    TensorFlow/Keras for building the ANN.
    Matplotlib and Seaborn for visualizations.
    Scikit-learn for preprocessing and evaluation metrics.

Data Preprocessing

Steps:

    Encoding categorical variables:

    Gender (binary encoding).
    Country (one-hot encoding).

    Splitting the dataset into training and test sets.
    Standardizing features to improve model performance.

#Example preprocessing code snippet
X = data.iloc[:, 3:-1].values
Y = data.iloc[:, -1].values

#Encoding categorical variables
LE1 = LabelEncoder()
X[:, 2] = np.array(LE1.fit_transform(X[:, 2]))

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

Model Training

A feedforward ANN was built with:

    Two hidden layers using ReLU activation.
    A sigmoid-activated output layer for binary classification.

# Model architecture
ann = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=6, activation="relu"),
    tf.keras.layers.Dense(units=6, activation="relu"),
    tf.keras.layers.Dense(units=1, activation="sigmoid")
])

# Compiling and training the ANN
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])
ann.fit(X_train, Y_train, batch_size=32, epochs=10)

Visualizations

Several visualizations were created to evaluate the model:

1. Loss and Accuracy Plots Displays the model's training progress over epochs.

plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.title('Model Performance')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.legend()
plt.show()
