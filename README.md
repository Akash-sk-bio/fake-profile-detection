# fake-profile-detection
This project implements a Convolutional Neural Network (CNN) to detect fake Instagram profiles based on numerical profile features such as the number of followers, following, and posts. The model has been trained to classify profiles as either Fake or Real by identifying patterns in these features.

### **Detailed Explanation of the CNN Architecture and Fake vs Real Profile Prediction**

#### **CNN Architecture Overview:**

The Convolutional Neural Network (CNN) we are using for detecting fake Instagram profiles consists of the following key components:

1. **Input Layer**:  
   - The input layer takes features like the number of followers, following, posts, and possibly other numerical data about the profile.
   - Each feature is scaled and reshaped to fit the CNN input, ensuring the network can process it properly.

2. **1D Convolutional Layers**:
   - These layers are responsible for extracting important features from the input data.
   - Each convolutional layer applies filters (kernels) over the input, detecting patterns in the data. In the context of social media profiles, patterns could include relationships between followers, following, and posts that signify a "fake" or "real" profile.
   - The **activation function** used here is `ReLU` (Rectified Linear Unit), which helps introduce non-linearity into the network.

3. **Pooling Layers**:
   - **Max Pooling** is used after each convolutional layer to downsample the feature maps, reducing their dimensions and computational cost. This helps the network focus on the most relevant patterns detected by the filters.

4. **Flatten Layer**:
   - After the final convolutional and pooling layers, the data is flattened into a 1D array. This flattened data is fed into the fully connected layers.

5. **Fully Connected Layers (Dense Layers)**:
   - These layers take the output from the convolutional layers and further learn the relationships between the features.
   - The network learns complex patterns such as whether a high follower-to-following ratio or an unusually high number of posts might indicate a fake profile.
   - The last dense layer has a single unit with a `sigmoid` activation function, which outputs a probability between 0 and 1.

6. **Output Layer**:
   - The output layer predicts whether the Instagram profile is "Fake" or "Real" based on the learned patterns.
   - A probability greater than 0.5 indicates a "Real" profile, while a probability less than 0.5 indicates a "Fake" profile.

#### **Training the Model:**
- The CNN is trained using the **binary cross-entropy loss** function, which is well-suited for binary classification tasks (fake vs real profiles).
- The optimizer used is **Adam**, which is a powerful and efficient optimization algorithm for deep learning.
- During training, the model adjusts its weights to minimize the error in prediction, learning to correctly classify profiles based on input features.

### **How the Model Predicts Fake vs Real Profiles:**

1. **Data Preprocessing**: The user inputs data such as the number of followers, following, and posts. These features are normalized (scaled) and reshaped to match the CNN input format.

2. **Convolution and Feature Extraction**: The CNN applies multiple filters over the input data. These filters detect patterns that could indicate if a profile is fake (e.g., an unusually high ratio of following to followers).

3. **Pooling and Dimensionality Reduction**: The pooling layers reduce the size of the feature maps, focusing on the most prominent patterns.

4. **Dense Layer Processing**: The fully connected layers learn deeper relationships between the features, further refining the ability to distinguish between fake and real profiles.

5. **Prediction Output**: Finally, the network outputs a probability. If the probability is above 0.5, the profile is predicted to be "Real"; if below 0.5, the profile is predicted to be "Fake."

---

### **README File**

```
# Fake Instagram Profile Detection Using CNN

## Project Overview
This project uses a Convolutional Neural Network (CNN) to classify Instagram profiles as either **Fake** or **Real** based on key profile features such as the number of followers, following, and posts. The model learns patterns from these features that help it detect potential fake accounts.

## CNN Model Architecture
- **Input Layer**: Takes features such as followers, following, and posts.
- **1D Convolutional Layers**: Extract feature patterns from the input data.
- **Max Pooling Layers**: Reduce dimensionality while retaining the most important information.
- **Fully Connected Layers (Dense Layers)**: Learn relationships between features.
- **Output Layer**: Outputs a probability indicating whether the profile is fake or real.

### How the CNN Model Works
1. **User Input**: Users enter the number of followers, following, and posts into the system.
2. **Feature Scaling**: The features are normalized and prepared for input into the CNN.
3. **Prediction**: The trained CNN model analyzes the input and predicts whether the profile is fake or real based on learned patterns.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/fake-profile-detection.git
cd fake-profile-detection
```

### 2. Install Dependencies
Make sure to install all the required dependencies by running:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should contain the following libraries:
- Django
- TensorFlow
- NumPy
- Pandas
- scikit-learn

### 3. Add the Pre-trained Model
Ensure the pre-trained model (`fake_profile_cnn_model.h5`) is placed in the `detection/` directory.

### 4. Set Up the Django Application
Once the project is cloned and dependencies are installed, set up the Django application.

```bash
python manage.py migrate
python manage.py runserver
```

### 5. Access the Application
Navigate to `http://127.0.0.1:8000/detection/predict/` in your browser. You will see a form where you can enter Instagram profile data (followers, following, posts).

### 6. Test the Model
After entering the data, the application will display whether the profile is predicted to be **Fake** or **Real**.

## Usage
### Input Fields:
- **Followers**: Enter the number of followers the profile has.
- **Following**: Enter the number of accounts the profile is following.
- **Posts**: Enter the number of posts made by the profile.

### Prediction:
- The model will output a prediction based on the input data:
    - **Fake**: The profile is predicted to be a fake account.
    - **Real**: The profile is predicted to be a real account.

## Model Explanation
The model uses a CNN to classify Instagram profiles based on several numerical features. It applies convolutional layers to extract meaningful patterns from the data and uses fully connected layers to make final predictions.

## Contributing
If you’d like to contribute to this project, feel free to submit a pull request.

## License
This project is licensed under the MIT License.
```

---

### **Additional Notes**:
1. **README Structure**: It contains setup instructions, model overview, and a guide on how to use the application and model for predictions.
2. **CNN Architecture Explanation**: A detailed explanation of how the CNN model processes the input and predicts fake or real profiles is included.

This setup should give you everything you need to document your CNN model and Django application for Instagram profile detection. Let me know if you need further assistance!
