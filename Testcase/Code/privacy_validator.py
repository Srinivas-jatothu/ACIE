# import logging
# import sys

# # --- Library Imports ---
# try:
#     import numpy as np
#     from numpy.linalg import norm
#     import torch
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'numpy' or 'torch' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install numpy torch")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Task 2.1 Import: The AI "Brain" ---
# try:
#     from model_loader import load_ai_model
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'model_loader.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Task 2.3 Import: The "Privacy Guard" ---
# try:
#     from privacy_guard import add_laplacian_noise, EPSILON, SENSITIVITY
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'privacy_guard.py' not found.")
#     logging.error("Please make sure 'privacy_guard.py' is in the same directory.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [PRIVACY_VALIDATOR] - %(message)s'
# )
# # --- End Configuration ---


# def generate_vector(text, model, tokenizer):
#     """
#     Generates a single, non-noisy vector (NumPy array) from a text string.
#     """
#     try:
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # Return the initial, non-noisy vector as a NumPy array
#         return outputs.pooler_output[0].numpy()
#     except Exception as e:
#         logging.error(f"Error generating vector for '{text}': {e}")
#         return None

# def cosine_similarity(v1, v2):
#     """
#     Calculates the cosine similarity between two NumPy vectors.
#     """
#     if v1 is None or v2 is None:
#         return 0.0
#     # Cosine Similarity = dot_product(A, B) / (norm(A) * norm(B))
#     return np.dot(v1, v2) / (norm(v1) * norm(v2))


# def main():
#     """
#     Runs the full privacy validation pipeline.
#     """
#     logging.info("--- [ Phase 5: Running Privacy Validation ] ---")

#     # 1. Load the AI "brain" (from Task 2.1)
#     model, tokenizer = load_ai_model()
#     if not model or not tokenizer:
#         logging.fatal("Failed to load AI model. Exiting.")
#         sys.exit(1)
        
#     logging.info(f"Privacy Parameters: Epsilon={EPSILON}, Sensitivity={SENSITIVITY}")

#     # --- Test 1: Anonymization Effectiveness Test ---
#     # This test proves that anonymizing a "secret" makes it
#     # mathematically distinct from the original, especially after noise.
    
#     logging.info("\n--- [ Test 1: Anonymization Effectiveness Test ] ---")
    
#     # The "attacker's" guess at a secret
#     secret_text = "process payment using sk_live_12345ABCDEF_real_key"
#     # The "anonymized" version our system *should* be using
#     anonymized_text = "process payment using [GATEWAY_API_KEY]"

#     # 1a. Generate initial (non-noisy) vectors
#     v_secret_initial = generate_vector(secret_text, model, tokenizer)
#     v_anonymized_initial = generate_vector(anonymized_text, model, tokenizer)
    
#     # 1b. Generate final (noisy) vectors
#     v_secret_noisy = add_laplacian_noise(v_secret_initial)
#     v_anonymized_noisy = add_laplacian_noise(v_anonymized_initial)

#     # 1c. Calculate similarities
#     sim_initial = cosine_similarity(v_secret_initial, v_anonymized_initial)
#     sim_noisy = cosine_similarity(v_secret_noisy, v_anonymized_noisy)

#     logging.info(f"Similarity between 'secret' and 'anonymized' (BEFORE noise): {sim_initial:.4f}")
#     logging.info(f"Similarity between 'secret' and 'anonymized' (AFTER noise):  {sim_noisy:.4f}")

#     test_1_passed = sim_noisy < (sim_initial * 0.9) # Check if noise significantly reduced similarity
#     logging.info(f"Test 1 Result: {'PASSED' if test_1_passed else 'FAILED'}")
#     if test_1_passed:
#         logging.info("  > SUCCESS: Noise & anonymization made the secret vector distinct.")
#     else:
#         logging.info("  > FAILURE: The noisy secret is still too similar to the noisy anonymized vector.")


#     # --- Test 2: Differential Privacy Noise Test ---
#     # This test proves that even with the *exact same* input, two
#     # different noisy vectors are generated. This stops an attacker
#     # from just re-generating our vector database.
    
#     logging.info("\n--- [ Test 2: Differential Privacy Noise Test ] ---")
    
#     base_text = "This is a base summary for testing."
    
#     # 2a. Generate one initial vector
#     v_initial = generate_vector(base_text, model, tokenizer)
    
#     # 2b. Add noise to it TWICE (this simulates two different runs)
#     v_noisy_A = add_laplacian_noise(v_initial)
#     v_noisy_B = add_laplacian_noise(v_initial)
    
#     # 2c. Calculate similarity
#     sim_self = cosine_similarity(v_noisy_A, v_noisy_B)
#     sim_perfect = cosine_similarity(v_noisy_A, v_noisy_A) # This will be 1.0

#     logging.info(f"Perfect similarity (vector vs. itself): {sim_perfect:.4f}")
#     logging.info(f"Similarity (noisy vector A vs. noisy vector B): {sim_self:.4f}")
    
#     test_2_passed = sim_self < 0.999 # Should not be 1.0
#     logging.info(f"Test 2 Result: {'PASSED' if test_2_passed else 'FAILED'}")
#     if test_2_passed:
#         logging.info("  > SUCCESS: Two noisy vectors from the same source are different.")
#     else:
#         logging.info("  > FAILURE: Noise is not being applied, or is not random.")


#     # --- Final Summary ---
#     logging.info("\n--- [ Validation Summary ] ---")
#     if test_1_passed and test_2_passed:
#         logging.info("OVERALL: PASSED")
#         logging.info("Your privacy mechanisms are working as expected.")
#     else:
#         logging.info("OVERALL: FAILED")
#         logging.info("One or more privacy validation tests failed. Review logs.")

# # --- Main Execution ---
# if __name__ == "__main__":
#     main()




# import logging
# import sys
# import json # Added to load project data

# # --- Library Imports ---
# try:
#     import numpy as np
#     from numpy.linalg import norm
#     import torch
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'numpy' or 'torch' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install numpy torch")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Task 2.1 Import: The AI "Brain" ---
# try:
#     from model_loader import load_ai_model
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'model_loader.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Task 2.3 Import: The "Privacy Guard" ---
# try:
#     from privacy_guard import add_laplacian_noise, EPSILON, SENSITIVITY
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'privacy_guard.py' not found.")
#     logging.error("Please make sure 'privacy_guard.py' is in the same directory.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [PRIVACY_VALIDATOR] - %(message)s'
# )
# INPUT_SUMMARIES_FILE = 'anonymized_summaries.json' # Added for Test 2
# # --- End Configuration ---


# def generate_vector(text, model, tokenizer):
#     """
#     Generates a single, non-noisy vector (NumPy array) from a text string.
#     """
#     try:
#         inputs = tokenizer(
#             text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # Return the initial, non-noisy vector as a NumPy array
#         return outputs.pooler_output[0].numpy()
#     except Exception as e:
#         logging.error(f"Error generating vector for '{text}': {e}")
#         return None

# def cosine_similarity(v1, v2):
#     """
#     Calculates the cosine similarity between two NumPy vectors.
#     """
#     if v1 is None or v2 is None:
#         return 0.0
#     # Cosine Similarity = dot_product(A, B) / (norm(A) * norm(B))
#     return np.dot(v1, v2) / (norm(v1) * norm(v2))

# def load_summaries(filename):
#     """
#     Loads the anonymized summaries JSON file.
#     """
#     logging.info(f"Loading summaries from '{filename}' for Test 2...")
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         return data
#     except FileNotFoundError:
#         logging.error(f"FATAL: Input file '{filename}' not found for Test 2.")
#         logging.error("Please run 'summarizer.py' (test.py) first to generate this file.")
#         return None
#     except json.JSONDecodeError:
#         logging.error(f"FATAL: Could not parse JSON from '{filename}'.")
#         return None


# def main():
#     """
#     Runs the full privacy validation pipeline.
#     """
#     logging.info("--- [ Phase 5: Running Privacy Validation ] ---")

#     # 1. Load the AI "brain" (from Task 2.1)
#     model, tokenizer = load_ai_model()
#     if not model or not tokenizer:
#         logging.fatal("Failed to load AI model. Exiting.")
#         sys.exit(1)
        
#     logging.info(f"Privacy Parameters: Epsilon={EPSILON}, Sensitivity={SENSITIVITY}")

#     # --- Test 1: Anonymization Effectiveness Test ---
#     # This test proves that anonymizing a "secret" makes it
#     # This test remains a controlled experiment with dummy text
#     # to ensure we have a "dirty" secret to test against.
    
#     logging.info("\n--- [ Test 1: Anonymization Effectiveness Test (Controlled) ] ---")
    
#     logging.info("\n--- [ Test 1: Anonymization Effectiveness Test ] ---")
    
#     # The "attacker's" guess at a secret
#     secret_text = "process payment using sk_live_12345ABCDEF_real_key"
#     # The "anonymized" version our system *should* be using
#     anonymized_text = "process payment using [GATEWAY_API_KEY]"

#     # 1a. Generate initial (non-noisy) vectors
#     v_secret_initial = generate_vector(secret_text, model, tokenizer)
#     v_anonymized_initial = generate_vector(anonymized_text, model, tokenizer)
    
#     # 1b. Generate final (noisy) vectors
#     v_secret_noisy = add_laplacian_noise(v_secret_initial)
#     v_anonymized_noisy = add_laplacian_noise(v_anonymized_initial)

#     # 1c. Calculate similarities
#     sim_initial = cosine_similarity(v_secret_initial, v_anonymized_initial)
#     sim_noisy = cosine_similarity(v_secret_noisy, v_anonymized_noisy)

#     logging.info(f"Similarity between 'secret' and 'anonymized' (BEFORE noise): {sim_initial:.4f}")
#     logging.info(f"Similarity between 'secret' and 'anonymized' (AFTER noise):  {sim_noisy:.4f}")

#     test_1_passed = sim_noisy < (sim_initial * 0.9) # Check if noise significantly reduced similarity
#     logging.info(f"Test 1 Result: {'PASSED' if test_1_passed else 'FAILED'}")
#     if test_1_passed:
#         logging.info("  > SUCCESS: Noise & anonymization made the secret vector distinct.")
#     else:
#         logging.info("  > FAILURE: The noisy secret is still too similar to the noisy anonymized vector.")


#     # --- Test 2: Differential Privacy Noise Test (On Real Project Data) ---
#     logging.info("\n--- [ Test 2: Differential Privacy Noise Test (On Real Project Data) ] ---")
    
#     # 1. Load real project data
#     summaries = load_summaries(INPUT_SUMMARIES_FILE)
#     base_text = None
    
#     if summaries:
#         try:
#             # Find the payment_service file (key ends with it)
#             payment_file_key = next(k for k in summaries if k.endswith('payment_service.py'))
#             # Get the 'process_payment' summary (last node in that file's list)
#             base_text = summaries[payment_file_key][-1]["summary"]
#             logging.info(f"Using real project summary for Test 2: '{base_text[:70]}...'")
#         except Exception as e:
#             logging.warning(f"Could not find specific project summary. Falling back to dummy text. Error: {e}")
            
#     if base_text is None:
#         # Fallback if file or node is missing
#         base_text = "This is a base summary for testing."
#         logging.info("Using dummy text for Test 2.")
    
#     # 2a. Generate one initial vector
#     v_initial = generate_vector(base_text, model, tokenizer)
    
#     if v_initial is None:
#         logging.error("Test 2 FAILED: Could not generate vector for base text.")
#         test_2_passed = False
#     else:
#         # 2b. Add noise to it TWICE (this simulates two different runs)
#         v_noisy_A = add_laplacian_noise(v_initial)
#         v_noisy_B = add_laplacian_noise(v_initial)
        
#         # 2c. Calculate similarity
#         sim_self = cosine_similarity(v_noisy_A, v_noisy_B)
#         sim_perfect = cosine_similarity(v_noisy_A, v_noisy_A) # This will be 1.0

#         logging.info(f"Perfect similarity (vector vs. itself): {sim_perfect:.4f}")
#         logging.info(f"Similarity (noisy vector A vs. noisy vector B): {sim_self:.4f}")
        
#         test_2_passed = sim_self < 0.999 # Should not be 1.0
#         logging.info(f"Test 2 Result: {'PASSED' if test_2_passed else 'FAILED'}")
#         if test_2_passed:
#             logging.info("  > SUCCESS: Two noisy vectors from the same source are different.")
#         else:
#             logging.info("  > FAILURE: Noise is not being applied, or is not random.")


#     # --- Final Summary ---
#     logging.info("\n--- [ Validation Summary ] ---")
#     if test_1_passed and test_2_passed:
#         logging.info("OVERALL: PASSED")
#         logging.info("Your privacy mechanisms are working as expected.")
#     else:
#         logging.info("OVERALL: FAILED")
#         logging.info("One or more privacy validation tests failed. Review logs.")

# # --- Main Execution ---
# if __name__ == "__main__":
#     main()


import logging
import sys
import json # Added to load project data

# --- Library Imports ---
try:
    import numpy as np
    from numpy.linalg import norm
    import torch
except ImportError:
    logging.basicConfig()
    logging.error("="*60)
    logging.error("FATAL: 'numpy' or 'torch' library not found.")
    logging.error("Please install them by running:")
    logging.error("pip install numpy torch")
    logging.error("="*60)
    sys.exit(1)

# --- Task 2.1 Import: The AI "Brain" ---
try:
    from model_loader import load_ai_model
except ImportError:
    logging.basicConfig()
    logging.error("="*60)
    logging.error("FATAL: 'model_loader.py' not found.")
    logging.error("Please make sure all .py scripts are in the same directory.")
    logging.error("="*60)
    sys.exit(1)

# --- Task 2.3 Import: The "Privacy Guard" ---
try:
    from privacy_guard import add_laplacian_noise, EPSILON, SENSITIVITY
except ImportError:
    logging.basicConfig()
    logging.error("="*60)
    logging.error("FATAL: 'privacy_guard.py' not found.")
    logging.error("Please make sure 'privacy_guard.py' is in the same directory.")
    logging.error("="*60)
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [PRIVACY_VALIDATOR] - %(message)s'
)
INPUT_SUMMARIES_FILE = 'anonymized_summaries.json' # Added for Test 2
# --- End Configuration ---


def generate_vector(text, model, tokenizer):
    """
    Generates a single, non-noisy vector (NumPy array) from a text string.
    """
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Return the initial, non-noisy vector as a NumPy array
        return outputs.pooler_output[0].numpy()
    except Exception as e:
        logging.error(f"Error generating vector for '{text}': {e}")
        return None

def cosine_similarity(v1, v2):
    """
    Calculates the cosine similarity between two NumPy vectors.
    """
    if v1 is None or v2 is None:
        return 0.0
    # Cosine Similarity = dot_product(A, B) / (norm(A) * norm(B))
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def load_summaries(filename):
    """
    Loads the anonymized summaries JSON file.
    """
    logging.info(f"Loading summaries from '{filename}' for Test 2...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logging.error(f"FATAL: Input file '{filename}' not found for Test 2.")
        logging.error("Please run 'summarizer.py' (test.py) first to generate this file.")
        return None
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not parse JSON from '{filename}'.")
        return None


def main():
    """
    Runs the full privacy validation pipeline.
    """
    logging.info("--- [ Phase 5: Running Privacy Validation ] ---")

    # 1. Load the AI "brain" (from Task 2.1)
    model, tokenizer = load_ai_model()
    if not model or not tokenizer:
        logging.fatal("Failed to load AI model. Exiting.")
        sys.exit(1)
        
    logging.info(f"Privacy Parameters: Epsilon={EPSILON}, Sensitivity={SENSITIVITY}")

    # --- Test 1: Anonymization Effectiveness Test ---
    # This test proves that anonymizing a "secret" makes it
    # This test remains a controlled experiment with dummy text
    # to ensure we have a "dirty" secret to test against.
    
    logging.info("\n--- [ Test 1: Anonymization Effectiveness Test (Controlled) ] ---")
    
    # The "attacker's" guess at a secret
    secret_text = "process payment using sk_live_12345ABCDEF_real_key"
    # The "anonymized" version our system *should* be using
    anonymized_text = "process payment using [GATEWAY_API_KEY]"

    # 1a. Generate initial (non-noisy) vectors
    v_secret_initial = generate_vector(secret_text, model, tokenizer)
    v_anonymized_initial = generate_vector(anonymized_text, model, tokenizer)
    
    # 1b. Generate final (noisy) vectors
    v_secret_noisy = add_laplacian_noise(v_secret_initial)
    v_anonymized_noisy = add_laplacian_noise(v_anonymized_initial)

    # 1c. Calculate similarities
    sim_initial = cosine_similarity(v_secret_initial, v_anonymized_initial)
    sim_noisy = cosine_similarity(v_secret_noisy, v_anonymized_noisy)

    logging.info(f"Similarity between 'secret' and 'anonymized' (BEFORE noise): {sim_initial:.4f}")
    logging.info(f"Similarity between 'secret' and 'anonymized' (AFTER noise):  {sim_noisy:.4f}")

    test_1_passed = sim_noisy < (sim_initial * 0.9) # Check if noise significantly reduced similarity
    logging.info(f"Test 1 Result: {'PASSED' if test_1_passed else 'FAILED'}")
    if test_1_passed:
        logging.info("  > SUCCESS: Noise & anonymization made the secret vector distinct.")
    else:
        logging.info("  > FAILURE: The noisy secret is still too similar to the noisy anonymized vector.")


    # --- Test 2: Differential Privacy Noise Test (On Real Project Data) ---
    logging.info("\n--- [ Test 2: Differential Privacy Noise Test (On Whole Project) ] ---")
    
    # 1. Load real project data
    summaries = load_summaries(INPUT_SUMMARIES_FILE)
    test_2_passed = False # Default to fail
    summaries_tested_count = 0
    
    if summaries:
        try:
            test_2_passed = True # Assume pass until a failure is found
            for file_path, nodes in summaries.items():
                if not test_2_passed: break # Stop if test has already failed
                
                for node_info in nodes:
                    base_text = node_info.get("summary")
                    node_name = node_info.get("name", "Unnamed Node")
                    
                    if not base_text:
                        logging.warning(f"Skipping node '{node_name}' in '{file_path}' (no summary text).")
                        continue
                    
                    summaries_tested_count += 1
                    
                    # 2a. Generate one initial vector
                    v_initial = generate_vector(base_text, model, tokenizer)
                    
                    if v_initial is None:
                        logging.error(f"Test 2 FAILED: Could not generate vector for node '{node_name}'.")
                        test_2_passed = False
                        break # Stop processing this file's nodes
                    
                    # 2b. Add noise to it TWICE
                    v_noisy_A = add_laplacian_noise(v_initial)
                    v_noisy_B = add_laplacian_noise(v_initial)
                    
                    # 2c. Calculate similarity
                    sim_self = cosine_similarity(v_noisy_A, v_noisy_B)
                    
                    if sim_self >= 0.999: # Similarity is 1.0 (or close enough)
                        logging.error(f"Test 2 FAILED for node '{node_name}'.")
                        logging.error("  > FAILURE: Noise is not being applied, or is not random (Similarity >= 0.999).")
                        test_2_passed = False
                        break # Stop processing this file's nodes
            
            if summaries_tested_count == 0:
                logging.warning("Test 2 FAILED: No summaries were found to test.")
                test_2_passed = False

        except Exception as e:
            logging.error(f"Test 2 FAILED due to an unexpected error: {e}")
            test_2_passed = False
            
    else:
        logging.error("Test 2 FAILED: Could not load summaries file.")
        test_2_passed = False
    
    # Log final Test 2 result
    logging.info(f"Test 2 Result: {'PASSED' if test_2_passed else 'FAILED'}")
    if test_2_passed:
        logging.info(f"  > SUCCESS: All {summaries_tested_count} summaries passed the differential privacy noise test.")
    else:
        logging.info("  > FAILURE: One or more summaries failed the test. Review logs.")


    # --- Final Summary ---
    logging.info("\n--- [ Validation Summary ] ---")
    if test_1_passed and test_2_passed:
        logging.info("OVERALL: PASSED")
        logging.info("Your privacy mechanisms are working as expected.")
    else:
        logging.info("OVERALL: FAILED")
        logging.info("One or more privacy validation tests failed. Review logs.")

# --- Main Execution ---
if __name__ == "__main__":
    main()

