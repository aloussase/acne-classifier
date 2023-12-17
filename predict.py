import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the saved model
model = load_model('acne_model.h5')

def live_predict():
    # Open a video capture
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
    
        # Preprocess the frame for prediction
        img = cv2.resize(frame, (32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
    
        # Make a prediction
        prediction = model.predict(img_array)
        print(f"Prediction: {prediction}")
    
        # Interpret the prediction
        if prediction[0, 0] > 0.5:
            label = 'Acne'
        else:
            label = 'No Acne'
    
        # Display the prediction
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Acne Classifier', frame)
    
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


def classify_path(img_path):
    img = image.load_img(img_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Interpret the prediction
    if prediction[0, 0] > 0.5:
        print("The image contains acne.")
    else:
        print("The image does not contain acne.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Acne Classifier")

    parser.add_argument('-l', '--live', action='store_true', help='Open a camera to predict live')
    parser.add_argument('-p', '--path', help='Path to an image to classify')

    args = parser.parse_args()

    if args.live:
        live_predict()
    elif args.path:
        classify_path(args.path)
    
