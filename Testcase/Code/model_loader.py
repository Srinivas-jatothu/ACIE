# import logging
# import sys

# # We try to import the key library.
# # If it's not found, we give a helpful error message and exit.
# try:
#     from transformers import AutoTokenizer, AutoModel
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'transformers' library not found.")
#     logging.error("Please install it by running: pip install transformers")
#     logging.error("You will also need PyTorch: pip install torch")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the main dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [MODEL_LOADER] - %(message)s'
# )

# # This is the "brain" we're loading, as specified in your document.
# # It's a model pre-trained on code,
# # ideal for both summarization and creating vector embeddings.
# MODEL_NAME = "microsoft/codebert-base"
# # --- End Configuration ---


# def load_ai_model(model_name=MODEL_NAME):
#     """
#     Downloads and loads the pre-trained AI model and tokenizer
#     from Hugging Face.
    
#     The files are cached locally after the first download.
#     """
#     logging.info(f"--- [ Task 2.1: Load Core AI Model ] ---")
#     logging.info(f"Attempting to load model: {model_name}")
    
#     try:
#         # 1. Load the Tokenizer
#         # The tokenizer's job is to convert raw code text
#         # into the numerical tokens the model understands.
#         logging.info(f"Loading tokenizer for {model_name}...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         logging.info("Tokenizer loaded successfully.")

#         # 2. Load the Model
#         # This is the "brain" itself. It's a large file
#         # and may take time to download on the first run.
#         logging.info(f"Loading model '{model_name}'...")
#         logging.info("(This may take a few minutes to download on first run...)")
#         model = AutoModel.from_pretrained(model_name)
#         logging.info("Model loaded successfully.")
        
#         return model, tokenizer

#     except OSError as e:
#         # This often happens if the model name is wrong or no internet
#         logging.error(f"Error loading model '{model_name}'.")
#         logging.error(f"Details: {e}")
#         logging.error("Please check your internet connection and the model name.")
#         return None, None
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return None, None

# if __name__ == "__main__":
    
#     # --- Main Execution ---
#     model, tokenizer = load_ai_model()
    
#     if model and tokenizer:
#         logging.info("\n--- [ AI Model Setup Complete ] ---")
#         logging.info(f"Model:    {model.__class__.__name__} (from {MODEL_NAME})")
#         logging.info(f"Tokenizer: {tokenizer.__class__.__name__} (from {MODEL_NAME})")
#         logging.info("The AI 'brain' is loaded and ready for Phase 3 (Summarization).")
#     else:
#         logging.info("\n--- [ AI Model Setup Failed ] ---")
#         logging.info("Please check the error messages above.")




# import logging
# import sys

# # We try to import the key library.
# # If it's not found, we give a helpful error message and exit.
# try:
#     from transformers import AutoTokenizer, AutoModel
#     import torch # Added torch import for the test function
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'transformers' or 'torch' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install transformers")
#     logging.error("pip install torch")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the main dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [MODEL_LOADER] - %(message)s'
# )

# # This is the "brain" we're loading, as specified in your document.
# # It's a model pre-trained on code,
# # ideal for both summarization and creating vector embeddings.
# MODEL_NAME = "microsoft/codebert-base"
# # --- End Configuration ---


# def load_ai_model(model_name=MODEL_NAME):
#     """
#     Downloads and loads the pre-trained AI model and tokenizer
#     from Hugging Face.
    
#     The files are cached locally after the first download.
#     """
#     logging.info(f"--- [ Task 2.1: Load Core AI Model ] ---")
#     logging.info(f"Attempting to load model: {model_name}")
    
#     try:
#         # 1. Load the Tokenizer
#         # The tokenizer's job is to convert raw code text
#         # into the numerical tokens the model understands.
#         logging.info(f"Loading tokenizer for {model_name}...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         logging.info("Tokenizer loaded successfully.")

#         # 2. Load the Model
#         # This is the "brain" itself. It's a large file
#         # and may take time to download on the first run.
#         logging.info(f"Loading model '{model_name}'...")
#         logging.info("(This may take a few minutes to download on first run...)")
#         model = AutoModel.from_pretrained(model_name)
#         logging.info("Model loaded successfully.")
        
#         return model, tokenizer

#     except OSError as e:
#         # This often happens if the model name is wrong or no internet
#         logging.error(f"Error loading model '{model_name}'.")
#         logging.error(f"Details: {e}")
#         logging.error("Please check your internet connection and the model name.")
#         return None, None
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return None, None

# # --- [ NEW TEST FUNCTION ] ---
# def test_model(model, tokenizer):
#     """
#     Runs a simple test to verify the model and tokenizer are working.
#     """
#     logging.info("\n--- [ Running Basic Model Test ] ---")
    
#     # A simple Python function to test with
#     sample_code = "def hello_world():\n    print('Hello from CodeBERT!')"

#     #print the summarization code snippet
#     logging.info(f"Test Code Snippet:\n{sample_code}")

#     #now we will tokenize and run through the model to see if it works correctly

    
#     try:
#         # 1. Tokenize the code
#         # return_tensors='pt' tells the tokenizer to return PyTorch tensors
#         inputs = tokenizer(sample_code, return_tensors="pt")
#         logging.info(f"Code tokenized successfully. Input tensor shape: {inputs['input_ids'].shape}")

#         # 2. Run tokens through the model
#         # We run this in a 'no_grad' block to save memory,
#         # as we are not training the model.
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # 'pooler_output' is a good representation of the [CLS] token,
#         # which represents the "meaning" of the whole snippet.
#         # Its shape should be [1, 768] (1 snippet, 768 dimensions)
#         pooled_output = outputs.pooler_output
        
#         logging.info(f"Model processed tokens successfully.")
#         logging.info(f"Output (pooled) shape: {pooled_output.shape}")
        
#         if pooled_output.shape == torch.Size([1, 768]):
#             logging.info("TEST PASSED: Model and tokenizer are working correctly.")
#         else:
#             logging.warning(f"TEST WARNING: Output shape {pooled_output.shape} is unexpected. Expected [1, 768].")

#     except Exception as e:
#         logging.error(f"--- TEST FAILED ---")
#         logging.error(f"An error occurred during the model test: {e}")
# # --- [ END NEW TEST FUNCTION ] ---



# def test_model_verbose(model, tokenizer):
#     """
#     Extended test function that shows:
#     1. Tokens generated by the tokenizer
#     2. Embedding vector from the model
#     """
#     logging.info("\n--- [ Running Verbose Model Test ] ---")
    
#     sample_code = "def hello_world():\n    print('Hello from CodeBERT!')"
#     logging.info(f"Test Code Snippet:\n{sample_code}\n")
    
#     # 1. Tokenize and show tokens
#     inputs = tokenizer(sample_code, return_tensors="pt")
#     tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
#     logging.info(f"Tokens ({len(tokens)}): {tokens}")
    
#     # 2. Run model
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     pooled_output = outputs.pooler_output
#     logging.info(f"Embedding vector (first 10 values): {pooled_output[0][:10]}")
#     logging.info(f"Full embedding shape: {pooled_output.shape}")
    
#     logging.info("\n✅ MODEL CHECK COMPLETE: Tokenization and embeddings are working.")





# if __name__ == "__main__":
    
#     # --- Main Execution ---
#     model, tokenizer = load_ai_model()
    
#     if model and tokenizer:
#         logging.info("\n--- [ AI Model Setup Complete ] ---")
#         logging.info(f"Model:    {model.__class__.__name__} (from {MODEL_NAME})")
#         logging.info(f"Tokenizer: {tokenizer.__class__.__name__} (from {MODEL_NAME})")
        
#         # --- Run the test ---
#         test_model(model, tokenizer)
        
#         logging.info("\nThe AI 'brain' is loaded and ready for Phase 3 (SummarIZATION).")
#     else:
#         logging.info("\n--- [ AI Model Setup Failed ] ---")
#         logging.info("Please check the error messages above.")




# import logging
# import sys
# import os

# # We try to import the key library.
# # If it's not found, we give a helpful error message and exit.
# try:
#     from transformers import AutoTokenizer, AutoModel
#     import torch # Added torch import for the test function
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'transformers' or 'torch' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install transformers")
#     logging.error("pip install torch")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the main dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [MODEL_LOADER] - %(message)s'
# )

# # This is the "brain" we're loading, as specified in your document.
# # It's a model pre-trained on code,
# # ideal for both summarization and creating vector embeddings.
# MODEL_NAME = "microsoft/codebert-base"
# # --- End Configuration ---


# def load_ai_model(model_name=MODEL_NAME):
#     """
#     Downloads and loads the pre-trained AI model and tokenizer
#     from Hugging Face.
    
#     The files are cached locally after the first download.
#     """
#     # ... (this function is unchanged) ...
#     logging.info(f"--- [ Task 2.1: Load Core AI Model ] ---")
#     logging.info(f"Attempting to load model: {model_name}")
    
#     try:
#         # 1. Load the Tokenizer
#         # The tokenizer's job is to convert raw code text
#         # into the numerical tokens the model understands.
#         logging.info(f"Loading tokenizer for {model_name}...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         logging.info("Tokenizer loaded successfully.")

#         # 2. Load the Model
#         # This is the "brain" itself. It's a large file
#         # and may take time to download on the first run.
#         logging.info(f"Loading model '{model_name}'...")
#         logging.info("(This may take a few minutes to download on first run...)")
#         model = AutoModel.from_pretrained(model_name)
#         logging.info("Model loaded successfully.")
        
#         return model, tokenizer

#     except OSError as e:
#         # ... (error handling is unchanged) ...
#         logging.error(f"Error loading model '{model_name}'.")
#         logging.error(f"Details: {e}")
#         logging.error("Please check your internet connection and the model name.")
#         return None, None
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")
#         return None, None

# # --- [ UPDATED TEST FUNCTION ] ---
# def test_model(model, tokenizer):
#     """
#     Runs a detailed test to verify the model and tokenizer,
#     and writes the results to 'codebert_testcase_result.txt'.
#     """
#     logging.info("\n--- [ Running Detailed Model Test ] ---")
    
#     # A simple Python function to test with
#     sample_code = "def hello_world():\n    print('Hello from CodeBERT!')"
#     output_filename = "codebert_testcase_result.txt"
    
#     # This list will store our formatted results
#     results_output = []
    
#     try:
#         results_output.append("--- [ CodeBERT Detailed Test Case Results ] ---")
#         results_output.append(f"Model: {MODEL_NAME}\n")
#         results_output.append("--- [ 1. Test Code Snippet ] ---")
#         results_output.append(f"{sample_code}\n")
        
#         # 1. Tokenize the code
#         inputs = tokenizer(sample_code, return_tensors="pt")
#         input_ids = inputs['input_ids'][0] # Get the token IDs for the first item
#         tokens = tokenizer.convert_ids_to_tokens(input_ids)

#         results_output.append("--- [ 2. Tokenization Results ] ---")
#         results_output.append(f"Input Tensor Shape (batch_size, num_tokens): {inputs['input_ids'].shape}")
#         results_output.append(f"\nToken IDs (Numerical Representation):\n{input_ids.tolist()}")
#         results_output.append(f"\nTokens (String Representation):\n{tokens}\n")

#         # 2. Run tokens through the model
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # 3. Analyze Model Functionalities
#         results_output.append("--- [ 3. Model Functionality Outputs ] ---")

#         # [A] Last Hidden State
#         last_hidden_state = outputs.last_hidden_state
#         results_output.append("\n[A] Last Hidden State (Vector for *each* token)")
#         results_output.append(f"  Shape (batch_size, num_tokens, hidden_dims): {last_hidden_state.shape}")
#         results_output.append("  This contains the 768-dimension vector for all 20 tokens.")
#         results_output.append(f"  Sample (First 5 dims of first token, '[CLS]'): {last_hidden_state[0, 0, :5].tolist()}...")
        
#         # [B] Pooler Output (The "Summarization")
#         pooled_output = outputs.pooler_output
#         results_output.append("\n[B] Pooler Output (Semantic 'Summarization' Vector)")
#         results_output.append("  This single vector represents the 'meaning' of the *entire* snippet.")
#         results_output.append(f"  Shape (batch_size, hidden_dims): {pooled_output.shape}")
#         results_output.append(f"  Sample (First 5 dims of summary vector): {pooled_output[0, :5].tolist()}...")

#         # 4. Final Test Verification
#         results_output.append("\n\n--- [ 4. Test Verification ] ---")
#         if pooled_output.shape == torch.Size([1, 768]) and last_hidden_state.shape[0] == 1:
#             test_result = "PASSED"
#             results_output.append("TEST PASSED: Model and tokenizer are working correctly.")
#             logging.info("TEST PASSED: Model and tokenizer are working correctly.")
#         else:
#             test_result = "FAILED"
#             results_output.append(f"TEST FAILED: Output shapes {pooled_output.shape} are unexpected.")
#             logging.warning(f"TEST FAILED: Output shapes {pooled_output.shape} are unexpected.")
            
#         # 5. Write results to file
#         try:
#             with open(output_filename, "w", encoding="utf-8") as f:
#                 f.write("\n".join(results_output))
#             logging.info(f"Successfully wrote detailed test results to: {output_filename}")
#         except IOError as e:
#             logging.error(f"Failed to write results file: {e}")
#             results_output.append(f"\nERROR: Failed to write results to file: {e}")

#     except Exception as e:
#         logging.error(f"--- TEST FAILED ---")
#         logging.error(f"An error occurred during the model test: {e}")
#         results_output.append(f"\n--- TEST FAILED ---\nError: {e}")
#         # Write failure log to file
#         try:
#             with open(output_filename, "w", encoding="utf-8") as f:
#                 f.write("\n".join(results_output))
#         except IOError:
#             pass # Ignore if file writing also fails
# # --- [ END UPDATED TEST FUNCTION ] ---


# if __name__ == "__main__":
    
#     # --- Main Execution ---
#     model, tokenizer = load_ai_model()
    
#     if model and tokenizer:
#         logging.info("\n--- [ AI Model Setup Complete ] ---")
#         logging.info(f"Model:    {model.__class__.__name__} (from {MODEL_NAME})")
#         logging.info(f"Tokenizer: {tokenizer.__class__.__name__} (from {MODEL_NAME})")
        
#         # --- Run the detailed test ---
#         test_model(model, tokenizer)
        
#         logging.info("\nThe AI 'brain' is loaded and ready for Phase 3 (Summarization).")
#     else:
#         logging.info("\n--- [ AI Model Setup Failed ] ---")
#         logging.info("Please check the error messages above.")





import logging
import sys
import os
import json # <-- NEW: To load the JSON file
import torch

# We try to import the key library.
# If it's not found, we give a helpful error message and exit.
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    logging.basicConfig() # Ensure logging is configured for the error
    logging.error("="*60)
    logging.error("FATAL: 'transformers' or 'torch' library not found.")
    logging.error("Please install them by running:")
    logging.error("pip install transformers")
    logging.error("pip install torch")
    logging.error("="*60)
    sys.exit(1) # Exit if the main dependency is missing

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [MODEL_LOADER] - %(message)s'
)

# This is the "brain" we're loading, as specified in your document.
MODEL_NAME = "microsoft/codebert-base"
HIERARCHICAL_NODES_FILE = "hierarchical_nodes.json" # <-- NEW: Path to our input
# --- End Configuration ---


def load_ai_model(model_name=MODEL_NAME):
    """
    Downloads and loads the pre-trained AI model and tokenizer
    from Hugging Face.
    
    The files are cached locally after the first download.
    """
    logging.info(f"--- [ Task 2.1: Load Core AI Model ] ---")
    logging.info(f"Attempting to load model: {model_name}")
    
    try:
        # 1. Load the Tokenizer
        logging.info(f"Loading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logging.info("Tokenizer loaded successfully.")

        # 2. Load the Model
        logging.info(f"Loading model '{model_name}'...")
        logging.info("(This may take a few minutes to download on first run...)")
        model = AutoModel.from_pretrained(model_name)
        logging.info("Model loaded successfully.")
        
        return model, tokenizer

    except OSError as e:
        logging.error(f"Error loading model '{model_name}'.")
        logging.error(f"Details: {e}")
        logging.error("Please check your internet connection and the model name.")
        return None, None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None, None

# --- [ UPDATED TEST FUNCTION ] ---
def test_model(model, tokenizer):
    """
    Runs a detailed test on the *first available code chunk*
    from the 'hierarchical_nodes.json' file.
    """
    logging.info(f"\n--- [ Running Detailed Model Test on First Extracted Chunk ] ---")
    
    # --- [ NEW: Get Real Code Chunk from JSON ] ---
    try:
        with open(HIERARCHICAL_NODES_FILE, "r", encoding="utf-8") as f:
            registry = json.load(f)
        
        if not registry:
            logging.error(f"'{HIERARCHICAL_NODES_FILE}' is empty. Cannot run test.")
            return

        # Find the very first node from the first file
        first_file_path = list(registry.keys())[0]
        first_node = registry[first_file_path][0]
        
        node_name = first_node['name']
        node_type = first_node['type']
        sample_code = first_node['anonymized_source_code'] # <-- Get the anonymized code
        
        logging.info(f"Using test input from: {first_file_path}")
        logging.info(f"Node: {node_type} '{node_name}'")
        
        # Create a dynamic output filename
        output_filename = f"codebert_test_result_{node_name}.txt"
        
    except FileNotFoundError:
        logging.error(f"FATAL: '{HIERARCHICAL_NODES_FILE}' not found.")
        logging.error(f"Please run 'node_extractor.py' first to generate this file.")
        return
    except Exception as e:
        logging.error(f"Failed to read or parse '{HIERARCHICAL_NODES_FILE}': {e}")
        return
    
    # Truncate for logging if chunk is too long
    if len(sample_code) > 500:
        logging_sample = sample_code[:500] + "\n... (truncated)"
    else:
        logging_sample = sample_code
    # --- [ END NEW: Get Real Code Chunk ] ---
    
    
    # This list will store our formatted results
    results_output = []
    
    try:
        results_output.append("--- [ CodeBERT Detailed Test Case Results ] ---")
        results_output.append(f"Model: {MODEL_NAME}")
        results_output.append(f"Test File: {first_file_path}")
        results_output.append(f"Test Node: {node_type} '{node_name}'\n")
        results_output.append("--- [ 1. Test Code Snippet (Anonymized) ] ---")
        results_output.append(f"{sample_code}\n") # Write full code to file
        
        # 1. Tokenize the code
        # We add truncation=True because a code chunk might be
        # longer than the model's 512-token limit.
        inputs = tokenizer(sample_code, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs['input_ids'][0] # Get the token IDs for the first item
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        results_output.append("--- [ 2. Tokenization Results ] ---")
        results_output.append(f"Input Tensor Shape (batch_size, num_tokens): {inputs['input_ids'].shape}")
        results_output.append(f"\nToken IDs (Numerical Representation):\n{input_ids.tolist()}")
        results_output.append(f"\nTokens (String Representation):\n{tokens}\n")

        # 2. Run tokens through the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 3. Analyze Model Functionalities
        results_output.append("--- [ 3. Model Functionality Outputs ] ---")

        # [A] Last Hidden State
        last_hidden_state = outputs.last_hidden_state
        results_output.append("\n[A] Last Hidden State (Vector for *each* token)")
        results_output.append(f"  Shape (batch_size, num_tokens, hidden_dims): {last_hidden_state.shape}")
        results_output.append("  This contains the 768-dimension vector for all tokens.")
        results_output.append(f"  Sample (First 5 dims of first token, '[CLS]'): {last_hidden_state[0, 0, :5].tolist()}...")
        
        # [B] Pooler Output (The "Summarization")
        pooled_output = outputs.pooler_output
        results_output.append("\n[B] Pooler Output (Semantic 'Summarization' Vector)")
        results_output.append("  This single vector represents the 'meaning' of the *entire* snippet.")
        results_output.append(f"  Shape (batch_size, hidden_dims): {pooled_output.shape}")
        results_output.append(f"  Sample (First 5 dims of summary vector): {pooled_output[0, :5].tolist()}...")

        # 4. Final Test Verification
        results_output.append("\n\n--- [ 4. Test Verification ] ---")
        # We check for 768 dims, but token count will vary based on file
        if pooled_output.shape == torch.Size([1, 768]) and last_hidden_state.shape[2] == 768:
            results_output.append("TEST PASSED: Model and tokenizer are working correctly.")
            logging.info("TEST PASSED: Model and tokenizer are working correctly.")
        else:
            results_output.append(f"TEST FAILED: Output shapes {pooled_output.shape} or {last_hidden_state.shape} are unexpected.")
            logging.warning(f"TEST FAILED: Output shapes {pooled_output.shape} are unexpected.")
            
        # 5. Write results to file
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(results_output))
            logging.info(f"Successfully wrote detailed test results to: {output_filename}")
        except IOError as e:
            logging.error(f"Failed to write results file: {e}")
            results_output.append(f"\nERROR: Failed to write results to file: {e}")

    except Exception as e:
        logging.error(f"--- TEST FAILED ---")
        logging.error(f"An error occurred during the model test: {e}")
        results_output.append(f"\n--- TEST FAILED ---\nError: {e}")
        # Write failure log to file
        try:
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n".join(results_output))
        except IOError:
            pass # Ignore if file writing also fails
# --- [ END UPDATED TEST FUNCTION ] ---


if __name__ == "__main__":
    
    # --- Main Execution ---
    model, tokenizer = load_ai_model()
    
    if model and tokenizer:
        logging.info("\n--- [ AI Model Setup Complete ] ---")
        logging.info(f"Model:    {model.__class__.__name__} (from {MODEL_NAME})")
        logging.info(f"Tokenizer: {tokenizer.__class__.__name__} (from {MODEL_NAME})")
        
        # --- Run the detailed test on a real, extracted code chunk ---
        test_model(model, tokenizer)
        
        logging.info("\nThe AI 'brain' is loaded and ready for Phase 3 (Summarization).")
    else:
        logging.info("\n--- [ AI Model Setup Failed ] ---")
        logging.info("Please check the error messages above.")

