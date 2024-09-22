Django Application
To integrate your CNN model for fake Instagram profile detection into a Django application and allow users to test the model with new data, follow these steps:
Steps for Building the Django Application:
1. Set up the Django Project:
First, install Django and other necessary dependencies (if not already installed):
pip install django tensorflow numpy pandas scikit-learn
Create a new Django project and app:
django-admin startproject fake_profile_detection
cd fake_profile_detection
python manage.py startapp detection
2. Save the Trained CNN Model:
Make sure your trained CNN model is saved as an .h5 file (e.g., fake_profile_cnn_model.h5), so you can load it in Django:
model.save('fake_profile_cnn_model.h5')
Place this .h5 file in a folder within your Django project (e.g., in the detection app directory).
3. Create the Model Prediction Logic:
In your detection app, create a file model.py to handle loading the saved model and preprocessing the input data for prediction.
detection/model.py:
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load the trained CNN model
model = tf.keras.models.load_model('detection/fake_profile_cnn_model.h5')

# Preprocess input (adjust this based on your input features)
def preprocess_input(data):
    # Convert input to a DataFrame for easier manipulation
    input_df = pd.DataFrame(data, index=[0])
    
    # Scale numerical features (assuming 'followers', 'following', 'posts')
    scaler = StandardScaler()
    numerical_cols = ['followers', 'following', 'posts']  # Adjust based on your columns
    input_df[numerical_cols] = scaler.fit_transform(input_df[numerical_cols])
    
    # Reshape input for CNN
    input_array = input_df.values.reshape((input_df.shape[0], input_df.shape[1], 1))
    
    return input_array

# Function to predict fake or real profile
def predict_profile(profile_data):
    processed_data = preprocess_input(profile_data)
    prediction = model.predict(processed_data)
    return 'Real' if prediction > 0.5 else 'Fake'
4. Create Views for the Prediction Form:
In detection/views.py, define a view that renders a form where users can input Instagram profile details, then call the CNN model to get the prediction.
detection/views.py:
from django.shortcuts import render
from .model import predict_profile

# View to handle profile prediction
def predict_view(request):
    if request.method == 'POST':
        # Get user input from the form
        profile_data = {
            'followers': int(request.POST['followers']),
            'following': int(request.POST['following']),
            'posts': int(request.POST['posts']),
        }

        # Predict using the CNN model
        prediction = predict_profile(profile_data)

        # Render the result page with the prediction
        return render(request, 'result.html', {'prediction': prediction})
    
    # Render the input form (if GET request)
    return render(request, 'predict.html')
5. Set up URLs:
In detection/urls.py, define the route for your prediction view.
detection/urls.py:
from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_view, name='predict'),
]
Then include the detection app's URLs in your project's main urls.py:
fake_profile_detection/urls.py:
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('detection/', include('detection.urls')),  # Include detection app URLs
]
6. Create HTML Templates:
Create a directory detection/templates and add two HTML files: predict.html (for the input form) and result.html (for the prediction result).
detection/templates/predict.html:
<!DOCTYPE html>
<html>
<head>
    <title>Fake Profile Detection</title>
</head>
<body>
    <h2>Instagram Profile Prediction</h2>
    <form method="POST">
        {% csrf_token %}
        <label for="followers">Number of Followers:</label>
        <input type="number" name="followers" required><br>

        <label for="following">Number of Following:</label>
        <input type="number" name="following" required><br>

        <label for="posts">Number of Posts:</label>
        <input type="number" name="posts" required><br>

        <input type="submit" value="Predict">
    </form>
</body>
</html>
detection/templates/result.html:

<!DOCTYPE html>
<html>
<head>
    <title>Prediction Result</title>
</head>
<body>
    <h2>Prediction Result</h2>
    <p>The Instagram profile is predicted to be: <strong>{{ prediction }}</strong></p>
    <a href="{% url 'predict' %}">Try another profile</a>
</body>
</html>
7. Run the Django Application:
Run the Django development server:
python manage.py runserver
Open your browser and navigate to http://127.0.0.1:8000/detection/predict/. You should see a form where you can input Instagram profile details (e.g., number of followers, following, posts), and upon submission, the app will return whether the profile is "Fake" or "Real."
Summary:
1.	Model Preparation: Load the trained CNN model and create preprocessing logic.
2.	Input Form: Create a form for users to input profile data.
3.	Prediction Logic: Handle user inputs and pass them through the CNN model to get predictions.
4.	Frontend: Use Django templates to display the input form and the result.
5.	Run the App: Serve the Django app to allow users to test the model.

