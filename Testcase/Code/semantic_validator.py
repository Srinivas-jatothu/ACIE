# import logging
# import sys
# import json
# import os

# # --- Library Imports ---
# try:
#     import numpy as np
#     import torch
#     import faiss
# except ImportError:
#     logging.basicConfig()
#     logging.error("=" * 60)
#     logging.error("FATAL: 'numpy', 'torch', or 'faiss-cpu' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install numpy torch faiss-cpu")
#     logging.error("=" * 60)
#     sys.exit(1)

# # --- Task 2.1 Import: The AI "Brain" ---
# try:
#     from model_loader import load_ai_model
# except ImportError:
#     logging.basicConfig()
#     logging.error("=" * 60)
#     logging.error("FATAL: 'model_loader.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("=" * 60)
#     sys.exit(1)

# # --- Task 2.3 Import: The "Privacy Guard" ---
# try:
#     from privacy_guard import add_laplacian_noise
# except ImportError:
#     logging.basicConfig()
#     logging.error("=" * 60)
#     logging.error("FATAL: 'privacy_guard.py' not found.")
#     logging.error("Please make sure 'privacy_guard.py' is in the same directory.")
#     logging.error("=" * 60)
#     sys.exit(1)

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [SEMANTIC_VALIDATOR] - %(message)s'
# )
# FAISS_INDEX_FILE = 'semantic_brain.faiss'
# MAPPING_FILE = 'semantic_brain_mapping.json'
# TOP_K_RESULTS = 3 # How many search results to check (e.g., is it in the top 3?)

# # --- Test Queries ---
# # This is the core of the validation. We define a query and the
# # "correct" answer we expect to see in the top results.
# TEST_QUERIES = [
#     {
#         "query": "How do I process a payment or transaction?",
#         "expected_top_result": "PaymentService.process_payment"
#     },
#     {
#         "query": "logic for calculating the shopping cart total price",
#         "expected_top_result": "CartService.calculate_total"
#     },
#     {
#         "query": "How do I validate a customer's cart?",
#         "expected_top_result": "CartService.validate_cart"
#     },
#     {
#         "query": "How is a final order created?",
#         "expected_top_result": "OrderService.finalize_order"
#     },
#     {
#         "query": "What object holds the user's items before purchase?",
#         "expected_top_result": "Cart.__init__"
#     }
# ]
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
#             padding=True,
#             max_length=512 # Ensure consistent padding
#         )
#         with torch.no_grad():
#             outputs = model(**inputs)
#         # Return the initial, non-noisy vector as a NumPy array
#         return outputs.pooler_output[0].numpy()
#     except Exception as e:
#         logging.error(f"Error generating vector for '{text}': {e}")
#         return None

# def load_faiss_index(index_path, mapping_path):
#     """
#     Loads the FAISS index and the ID-to-name mapping file.
#     """
#     logging.info(f"Loading FAISS index from '{index_path}'...")
#     try:
#         index = faiss.read_index(index_path)
#     except Exception as e:
#         logging.error(f"FATAL: Could not read FAISS index file '{index_path}'.")
#         logging.error(f"Error: {e}")
#         logging.error("Please run 'semantic_brain_builder.py' first.")
#         return None, None

#     logging.info(f"Loading mapping from '{mapping_path}'...")
#     try:
#         with open(mapping_path, 'r', encoding='utf-8') as f:
#             mapping = json.load(f)
#     except Exception as e:
#         logging.error(f"FATAL: Could not read mapping file '{mapping_path}'.")
#         logging.error(f"Error: {e}")
#         logging.error("Please run 'semantic_brain_builder.py' first.")
#         return None, None
        
#     logging.info("Successfully loaded 'Semantic Brain' and mapping.")
#     return index, mapping


# def main():
#     """
#     Runs the full semantic validation pipeline.
#     """
#     logging.info("--- [ Task 5.2: Running Semantic Validation ] ---")

#     # 1. Load the AI "brain" (to vectorize our queries)
#     model, tokenizer = load_ai_model()
#     if not model or not tokenizer:
#         logging.fatal("Failed to load AI model. Exiting.")
#         sys.exit(1)

#     # 2. Load the "Semantic Brain" (FAISS index + mapping)
#     index, mapping = load_faiss_index(FAISS_INDEX_FILE, MAPPING_FILE)
#     if not index or not mapping:
#         logging.fatal("Failed to load 'Semantic Brain'. Exiting.")
#         sys.exit(1)

#     all_tests_passed = True
#     logging.info(f"--- [ Running {len(TEST_QUERIES)} Semantic Test Queries ] ---")

#     for test in TEST_QUERIES:
#         query_text = test["query"]
#         expected_result = test["expected_top_result"]
#         logging.info(f"\nTESTING QUERY: \"{query_text}\"")
#         logging.info(f"  > EXPECTED: '{expected_result}' in top {TOP_K_RESULTS}")

#         # 3. Vectorize the query
#         query_vector_initial = generate_vector(query_text, model, tokenizer)
#         if query_vector_initial is None:
#             logging.error("  > FAILURE: Could not generate vector for query.")
#             all_tests_passed = False
#             continue

#         # 4. Apply the *same* privacy noise to our query
#         # We must search a "noisy" database with a "noisy" query
#         query_vector_noisy = add_laplacian_noise(query_vector_initial)
        
#         # FAISS expects a 2D array (batch of queries)
#         query_vector_2d = query_vector_noisy.reshape(1, -1)
        
#         # 5. Search the FAISS index
#         try:
#             # D = distances, I = indices (our internal IDs)
#             distances, indices = index.search(query_vector_2d.astype('float32'), TOP_K_RESULTS)
            
#             top_result_ids = indices[0] # Get the list of IDs
            
#             # 6. Map IDs back to human-readable names
#             top_result_names = []
#             for id_ in top_result_ids:
#                 if id_ == -1: continue # -1 means no result
#                 # Find the name from our mapping.json
#                 top_result_names.append(mapping.get(str(id_), "Unknown ID"))

#             logging.info(f"  > TOP {TOP_K_RESULTS} RESULTS: {top_result_names}")

#             # 7. Check for success
#             if expected_result in top_result_names:
#                 logging.info(f"  > PASS: Found expected result.")
#             else:
#                 logging.error(f"  > FAILURE: Expected result '{expected_result}' was NOT found.")
#                 all_tests_passed = False
                
#         except Exception as e:
#             logging.error(f"  > FAILURE: Error during FAISS search: {e}")
#             all_tests_passed = False

#     # --- Final Summary ---
#     logging.info("\n--- [ Semantic Validation Summary ] ---")
#     if all_tests_passed:
#         logging.info("OVERALL: PASSED")
#         logging.info("Your 'Semantic Brain' is providing useful and accurate results!")
#     else:
#         logging.info("OVERALL: FAILED")
#         logging.warning("One or more tests failed. This means your summaries (from smart_summarizer.py) "
#                         "may not be high-quality enough, or your test queries need tuning.")

# # --- Main Execution ---
# if __name__ == "__main__":
#     main()




import logging
import sys
import json
import os

# --- Library Imports ---
try:
    import numpy as np
    import torch
    import faiss
except ImportError:
    logging.basicConfig()
    logging.error("=" * 60)
    logging.error("FATAL: 'numpy', 'torch', or 'faiss-cpu' library not found.")
    logging.error("Please install them by running:")
    logging.error("pip install numpy torch faiss-cpu")
    logging.error("=" * 60)
    sys.exit(1)

# --- Task 2.1 Import: The AI "Brain" ---
try:
    # We will "run" the ast_parser's main block as a function
    # To do this, we need to refactor ast_parser.py slightly
    # (See instructions after this code block)
    
    # For now, let's assume we can import a "main" function
    # that returns the registry
    from model_loader import load_ai_model

except ImportError:
    logging.basicConfig()
    logging.error("=" * 60)
    logging.error("FATAL: 'model_loader.py' not found.")
    logging.error("Please make sure all .py scripts are in the same directory.")
    logging.error("=" * 60)
    sys.exit(1)

# --- Task 2.3 Import: The "Privacy Guard" ---
try:
    from privacy_guard import add_laplacian_noise
except ImportError:
    logging.basicConfig()
    logging.error("=" * 60)
    logging.error("FATAL: 'privacy_guard.py' not found.")
    logging.error("Please make sure 'privacy_guard.py' is in the same directory.")
    logging.error("=" * 60)
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [SEMANTIC_VALIDATOR] - %(message)s'
)
FAISS_INDEX_FILE = 'semantic_brain.faiss'
MAPPING_FILE = 'semantic_brain_mapping.json'
TOP_K_RESULTS = 3 # How many search results to check (e.g., is it in the top 3?)

# --- Test Queries ---
TEST_QUERIES = [
    {
        "query": "How do I process a payment or transaction?",
        "expected_top_result": "PaymentService.process_payment"
    },
    {
        "query": "logic for calculating the shopping cart total price",
        "expected_top_result": "CartService.calculate_total"
    },
    {
        "query": "How do I validate a customer's cart?",
        "expected_top_result": "CartService.validate_cart"
    },
    {
        "query": "How is a final order created?",
        "expected_top_result": "OrderService.finalize_order"
    },
    {
        "query": "What object holds the user's items before purchase?",
        "expected_top_result": "Cart.__init__"
    }
]
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
            padding=True,
            max_length=512 # Ensure consistent padding
        )
        with torch.no_grad():
            outputs = model(**inputs)
        # Return the initial, non-noisy vector as a NumPy array
        return outputs.pooler_output[0].numpy()
    except Exception as e:
        logging.error(f"Error generating vector for '{text}': {e}")
        return None

def load_faiss_index(index_path, mapping_path):
    """
    Loads the FAISS index and the ID-to-name mapping file.
    """
    logging.info(f"Loading FAISS index from '{index_path}'...")
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        logging.error(f"FATAL: Could not read FAISS index file '{index_path}'.")
        logging.error(f"Error: {e}")
        logging.error("Please run 'semantic_brain_builder.py' first.")
        return None, None

    logging.info(f"Loading mapping from '{mapping_path}'...")
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
    except Exception as e:
        logging.error(f"FATAL: Could not read mapping file '{mapping_path}'.")
        logging.error(f"Error: {e}")
        logging.error("Please run 'semantic_brain_builder.py' first.")
        return None, None
        
    logging.info("Successfully loaded 'Semantic Brain' and mapping.")
    return index, mapping


def main():
    """
    Runs the full semantic validation pipeline.
    """
    logging.info("--- [ Task 5.2: Running Semantic Validation ] ---")

    # 1. Load the AI "brain" (to vectorize our queries)
    model, tokenizer = load_ai_model()
    if not model or not tokenizer:
        logging.fatal("Failed to load AI model. Exiting.")
        sys.exit(1)

    # 2. Load the "Semantic Brain" (FAISS index + mapping)
    index, mapping = load_faiss_index(FAISS_INDEX_FILE, MAPPING_FILE)
    if not index or not mapping:
        logging.fatal("Failed to load 'Semantic Brain'. Exiting.")
        sys.exit(1)

    all_tests_passed = True
    logging.info(f"--- [ Running {len(TEST_QUERIES)} Semantic Test Queries ] ---")

    for test in TEST_QUERIES:
        query_text = test["query"]
        expected_result = test["expected_top_result"]
        logging.info(f"\nTESTING QUERY: \"{query_text}\"")
        logging.info(f"  > EXPECTED: '{expected_result}' in top {TOP_K_RESULTS}")

        # 3. Vectorize the query
        query_vector_initial = generate_vector(query_text, model, tokenizer)
        if query_vector_initial is None:
            logging.error("  > FAILURE: Could not generate vector for query.")
            all_tests_passed = False
            continue

        # 4. Apply the *same* privacy noise to our query
        query_vector_noisy = add_laplacian_noise(query_vector_initial)
        query_vector_2d = query_vector_noisy.reshape(1, -1)
        
        # 5. Search the FAISS index
        try:
            # D = distances, I = indices (our internal IDs)
            distances, indices = index.search(query_vector_2d.astype('float32'), TOP_K_RESULTS)
            
            top_result_ids = indices[0] # Get the list of IDs
            
            # 6. Map IDs back to human-readable names
            top_result_objects = []
            for id_ in top_result_ids:
                if id_ == -1: continue # -1 means no result
                top_result_objects.append(mapping.get(str(id_), "Unknown ID"))

            # --- [ THIS IS THE FIX ] ---
            # 7. Check for success by looking at the 'node_name' in the result objects
            # Extract just the names from the result dictionaries
            found_names = [
                result.get('node_name') for result in top_result_objects 
                if isinstance(result, dict)
            ]
            logging.info(f"  > TOP {TOP_K_RESULTS} RESULTS (Names): {found_names}")

            if expected_result in found_names:
                logging.info(f"  > PASS: Found expected result.")
            else:
                logging.error(f"  > FAILURE: Expected result '{expected_result}' was NOT found.")
                all_tests_passed = False
            # --- [ END OF FIX ] ---
                
        except Exception as e:
            logging.error(f"  > FAILURE: Error during FAISS search: {e}")
            all_tests_passed = False

    # --- Final Summary ---
    logging.info("\n--- [ Semantic Validation Summary ] ---")
    if all_tests_passed:
        logging.info("OVERALL: PASSED")
        logging.info("Your 'Semantic Brain' is providing useful and accurate results!")
    else:
        logging.info("OVERALL: FAILED")
        logging.warning("One or more tests failed. This means your summaries (from smart_summarizer.py) "
                        "may not be high-quality enough, or your test queries need tuning.")
        logging.warning("If you haven't run 'smart_summarizer.py' yet, do that now.")


# --- Main Execution ---
if __name__ == "__main__":
    main()

