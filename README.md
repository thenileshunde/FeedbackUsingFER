# FER - Facial Expression Recognition
This system generates feedback based on
real-time facial emotion recognition of the viewer and giving insights on the content review to
the creator, which aides he ameliorate his/her content and improves the viewers experience.


## Instructions
Files Structure:
- FER_CNN.ipynb - Tutorial to train the CNN
- ferMod.py - Uses the pre-trained model to give inferences
- model.json - Neural network architecture
- weights.h5 - Trained model weights
## Installation

* For model prediction

    `pip install -r requirements.txt`
    
    OR
    
    `pip install opencv-python`
    
    `pip install tensorflow` 
    
    `pip install keras`
    
* For model training,
    `pandas` `numpy` `tensorflow` `keras` `matplotlib` `scikit-learn` `seaborn`
    
* Running the inference engine

Use the webcam

`python FER.py webcam <fps>`

Use a video file

`python FER.py <video_file_name> <fps>`


