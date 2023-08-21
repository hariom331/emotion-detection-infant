# emotion-detection-infant
This project is intended to continuously monitor an infant and share the status of the infant with the caretaker through a wireless network. It captures sound and images and regularly monitors and send the resultant notification to the caretaker on regular intervals. 
# Database for Model Training
To access the image dataset, either write to "cityinfantfacedatabase@gmail.com" with credit for the dataset research or "hariomjoshi331@gmail.com" for either sound or both datasets. 
### Machine learning Models
In this repository the sound model is available but to be more precise you can retrain a fresh model by running the "python training.py" command in the sound process folder. And image model could not be made available due to contraints. However after putting the data in suitable folder heirarchy image model can also be trained by "python TrainEmotionDetector.py" in the Infant-Emotion-Detection folder. 
# Results
After successfully training the respective models the program can be run by pushing "python TestEmotionDetection.py" in infant folder and "python test.py" in infant-emotion-detection folder.
After That it starts recording through the device input devices, predicts output, and send notification to caretaker which address and information is to be updated in the code. 
# Project Demonstration
The demonstration of projects can be viewed on https://youtu.be/Ez0-x_NDGdQ and https://youtu.be/FJvehz0jm38 .
