from flask import Flask, request, jsonify
from transformers import pipeline
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Initialize the pipeline with the specified model
pipe = pipeline("image-to-text", model="lokibots/vit-patch16-1280-gpt2-large-image-summary")

@app.route('/api/recognize', methods=['POST'])
def recognize_artifact():
    # Check if the file is present in the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # Ensure a file was uploaded
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open and process the image
        image = Image.open(io.BytesIO(file.read()))

        # Use the pipeline to get artifact details
        result = pipe(image)

        # Extract the generated text from the result
        artifact_details = result[0]['generated_text']

        # Return the response
        return jsonify({
            "artifact_details": artifact_details
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
