from flask import Flask, request
from PIL import Image
from io import BytesIO
import base64
from joblib import load
from flask import Flask, request, jsonify

app = Flask(__name__)


# Function to load models
def load_model(model_type):
    if model_type == "svm":
        return load("models/svm.joblib")
    elif model_type == "tree":
        return load("models/DecisionTree.joblib")
    elif model_type == "lr":
        return load("models/B20ME036_lr_solver:saga.joblib")


def preprocess_image(base64_string, model):
    try:
        # Decode the base64 string and open it as an image
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))

        # Perform your image preprocessing here
        # For example, you can resize the image and convert it to grayscale
        image = image.resize((128, 128))
        image = image.convert("L")  # Convert to grayscale
        digit = model.predict(image)
        # Optionally, save the preprocessed image to a file
        # image.save('preprocessed_image.jpg')

        return digit
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        return None


@app.route("/predict", methods=["POST"])
def predict(model_type):
    model = load_model(model_type)

    try:
        data = request.get_json()
        if "image" in data and len(data["image"]) == 2:
            # Assuming data['image'] is a list of base64-encoded image strings
            image1_base64 = data["image"][0]
            image2_base64 = data["image"][1]

            op1 = preprocess_image(image1_base64, model)
            op2 = preprocess_image(image2_base64, model)

            # You can process the images here or save them to disk
            # For demonstration, we will print the lengths of the images
            print(f"Image 1 length: {len(image1_base64)}")
            print(f"Image 2 length: {len(image2_base64)}")
            if op1 == op2:
                return jsonify({"message": "Images are same."})
            else:
                return jsonify({"message": "Images are different."})
        else:
            return jsonify({"error": "Invalid data format."}), 400
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run()
