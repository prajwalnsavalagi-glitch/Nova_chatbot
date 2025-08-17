# Import necessary libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
# We now need the 'transformers' library to run the LLM
from transformers import pipeline

# Initialize the Flask app
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing)
CORS(app)

# ----------------------------------------------------
# We are switching back to the smaller, more efficient 'distilgpt2' model.
# This model is much easier to run and should fix the loading error we saw before.
print("Loading the language model. This will take a moment...")
llm_pipeline = None  # Initialize the pipeline as None outside the try block
try:
    # Use 'text-generation' pipeline for conversational AI
    # We are now using the 'distilgpt2' model from Hugging Face.
    llm_pipeline = pipeline('text-generation', model='distilgpt2')
    print("Model loaded successfully!")
except Exception as e:
    # This line is new! It will print the exact error to your terminal.
    print(f"An error occurred while loading the model: {e}")

# ----------------------------------------------------
# This is now the real function that gets a response from the LLM
def get_llm_response(prompt):
    """
    This function uses the loaded model to generate a response.
    """
    if not llm_pipeline:
        return "Sorry, the language model failed to load."

    try:
        # The pipeline returns a list of dictionaries, so we extract the text.
        # We also limit the maximum length of the response to keep it simple.
        response = llm_pipeline(
            prompt, 
            max_new_tokens=50, 
            num_return_sequences=1,
            # This makes sure the model doesn't just repeat the prompt.
            do_sample=True,
            temperature=0.7
        )
        # Extract the generated text from the first sequence
        generated_text = response[0]['generated_text']
        
        # The model might include your original prompt in the response.
        # We can clean this up by removing the prompt from the beginning.
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating a response from the model."


# Define the API endpoint that the frontend will call
@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles POST requests from the frontend for chat messages.
    """
    # Check if the request body is valid JSON
    if not request.json or 'prompt' not in request.json:
        return jsonify({'error': 'Invalid request body'}), 400

    # Get the user's prompt from the request
    prompt = request.json['prompt']

    # Get the response from your language model
    try:
        llm_response = get_llm_response(prompt)
        # Return the response as a JSON object
        return jsonify({'text': llm_response})
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# Run the Flask application
if __name__ == '__main__':
    # The server will run on http://localhost:5000
    print("Starting Flask server on http://localhost:5000...")
    print("This is the backend for your independent chatbot.")
    app.run(debug=True, port=5000)
