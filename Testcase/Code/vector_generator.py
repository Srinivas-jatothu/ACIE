# import logging
# import sys
# import json

# # --- Task 2.1 Import: The AI "Brain" ---
# try:
#     # We need the model and tokenizer to create the vectors
#     from model_loader import load_ai_model
#     import torch # PyTorch is needed for model inference
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'model_loader.py' or 'torch' library not found.")
#     logging.error("Please make sure all .py scripts and libraries are installed.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Task 2.3 Import: The "Privacy Guard" ---
# try:
#     # We need the function to add noise to our vectors
#     from privacy_guard import add_laplacian_noise
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'privacy_guard.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1)

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [VECTOR_GENERATOR] - %(message)s'
# )

# # This is the (Input) from Phase 3
# INPUT_SUMMARIES_FILE = 'anonymized_summaries.json'
# # This is the (Output) for Phase 4
# OUTPUT_VECTORS_FILE = 'privacy_preserving_vectors.json'
# # --- End Configuration ---


# def load_summaries(filename):
#     """
#     Loads the anonymized summaries JSON file (Output 1 from Phase 3).
#     """
#     logging.info(f"Loading anonymized summaries from '{filename}'...")
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logging.info(f"Successfully loaded {len(data)} files from summaries.")
#         return data
#     except FileNotFoundError:
#         logging.error(f"FATAL: Input file '{filename}' not found.")
#         logging.error("Please run 'summarizer.py' first to generate this file.")
#         return None
#     except json.JSONDecodeError:
#         logging.error(f"FATAL: Could not parse JSON from '{filename}'.")
#         return None

# def generate_initial_vector(summary_text, model, tokenizer):
#     """
#     Task 4.1: Generates an initial vector from a text summary.
#     This is the first layer of privacy (embedding the summary, not the code).
#     """
#     logging.debug(f"Generating vector for summary: '{summary_text[:60]}...'")
#     try:
#         # 1. Tokenize the summary text
#         inputs = tokenizer(
#             summary_text,
#             return_tensors="pt", # Return PyTorch tensors
#             truncation=True,     # Truncate long summaries
#             padding=True         # Pad to a uniform length
#         )
        
#         # 2. Feed tokens to the model to get embeddings
#         with torch.no_grad(): # Disable gradient calculation for efficiency
#             outputs = model(**inputs)
        
#         # 3. Get the "pooled" output
#         # This is a single 768-dimension vector that represents
#         # the "meaning" of the entire summary string.
#         # We take [0] because we process one summary at a time (batch size of 1)
#         initial_vector = outputs.pooler_output[0]
        
#         # 4. Convert from a PyTorch tensor to a NumPy array
#         # Our noise function (from privacy_guard.py) expects a NumPy array.
#         return initial_vector.numpy()
        
#     except Exception as e:
#         logging.error(f"Error generating vector for '{summary_text}': {e}")
#         return None

# def process_summaries_to_vectors(summaries_data, model, tokenizer):
#     """
#     Main orchestration function.
#     Iterates through all summaries, generates a base vector,
#     and applies differential privacy noise.
#     """
    
#     # This will store our final "Output 2"
#     vector_registry = {}
    
#     total_nodes = 0
#     for file_path, nodes in summaries_data.items():
#         file_vectors = [] # A list to hold all vectors for this file
        
#         logging.info(f"Processing {len(nodes)} nodes for: {file_path}")
        
#         for node in nodes:
#             summary = node.get('summary', '')
#             node_name = node.get('name', 'UnnamedNode')
#             node_type = node.get('type', 'UnknownType')
            
#             if not summary:
#                 logging.warning(f"Skipping {node_name}, no summary found.")
#                 continue

#             # --- Task 4.1: Generate Initial Vector ---
#             initial_vector = generate_initial_vector(summary, model, tokenizer)
            
#             if initial_vector is None:
#                 logging.warning(f"Failed to generate vector for {node_name}, skipping.")
#                 continue
                
#             logging.debug(f"  > Generated initial vector for {node_name}. Shape: {initial_vector.shape}")

#             # --- Task 2.3 / 4.2: Apply Differential Privacy Noise ---
#             # This is the second, mathematical layer of privacy.
#             noisy_vector = add_laplacian_noise(initial_vector)
#             logging.debug(f"  > Applied differential privacy noise to vector for {node_name}.")

#             # --- Store the Final, Secure Vector ---
#             # We convert the NumPy array to a plain list for JSON storage
#             file_vectors.append({
#                 'type': node_type,
#                 'name': node_name,
#                 'privacy_preserving_vector': noisy_vector.tolist()
#             })
#             total_nodes += 1
            
#         vector_registry[file_path] = file_vectors
        
#     logging.info(f"\nSuccessfully generated {total_nodes} privacy-preserving vectors.")
#     return vector_registry

# def save_vectors(registry, filename):
#     """
#     Saves the final vector registry to a JSON file.
#     """
#     logging.info(f"Exporting final vectors to '{filename}'...")
#     try:
#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(registry, f, indent=2)
#         logging.info(f"Successfully saved vectors.")
#     except (IOError, TypeError) as e:
#         logging.error(f"Failed to save vector file: {e}")


# # --- Main Execution ---
# if __name__ == "__main__":
    
#     logging.info("--- [ Phase 4: Generating Privacy-Preserving Vectors ] ---")

#     # 1. Load the AI "brain" (from Task 2.1)
#     model, tokenizer = load_ai_model()
#     if not model or not tokenizer:
#         logging.fatal("Failed to load AI model. Exiting.")
#         sys.exit(1)
        
#     # 2. Load the Anonymized Summaries (Output from Phase 3)
#     summaries = load_summaries(INPUT_SUMMARIES_FILE)
#     if not summaries:
#         logging.fatal("Failed to load summaries file. Exiting.")
#         sys.exit(1)

#     # 3. Run the full Phase 4 pipeline
#     # (Generate vectors and apply noise)
#     vector_registry = process_summaries_to_vectors(summaries, model, tokenizer)

#     # 4. Save the final "Output 2"
#     if vector_registry:
#         save_vectors(vector_registry, OUTPUT_VECTORS_FILE)
#         logging.info("\n--- [ Phase 4 Complete ] ---")
#         logging.info(f"Successfully created '{OUTPUT_VECTORS_FILE}'.")
#         logging.info("This file is 'Output 2' and is ready for the Vector Database.")
#     else:
#         logging.warning("No vectors were generated.")






# import logging
# import sys
# import json
# import os

# # --- Library Imports ---
# # Make sure to run: pip install faiss-cpu numpy torch transformers
# try:
#     import faiss
#     import numpy as np
#     import torch
# except ImportError:
#     logging.basicConfig()
#     logging.error("="*60)
#     logging.error("FATAL: 'faiss-cpu' or 'numpy' or 'torch' library not found.")
#     logging.error("Please install them by running:")
#     logging.error("pip install faiss-cpu numpy torch transformers")
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
#     from privacy_guard import add_laplacian_noise
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
#     format='%(asctime)s - %(levelname)s - [SEMANTIC_BRAIN_BUILDER] - %(message)s'
# )

# # This is the (Input) from Phase 3
# INPUT_SUMMARIES_FILE = 'anonymized_summaries.json'

# # --- Task 4.3: Output Files ---
# # This file will store the "Semantic Brain" index
# OUTPUT_INDEX_FILE = 'semantic_brain.faiss'
# # This file maps the index (e.g., ID=5) to its meaning ('payment_service.py', 'process_payment')
# OUTPUT_MAPPING_FILE = 'semantic_brain_mapping.json'

# # The dimension of our CodeBERT vectors
# VECTOR_DIMENSION = 768
# # --- End Configuration ---


# def load_summaries(filename):
#     """
#     Loads the anonymized summaries JSON file (Output 1 from Phase 3).
#     """
#     logging.info(f"Loading anonymized summaries from '{filename}'...")
#     try:
#         with open(filename, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logging.info(f"Successfully loaded {len(data)} files from summaries.")
#         return data
#     except FileNotFoundError:
#         logging.error(f"FATAL: Input file '{filename}' not found.")
#         logging.error("Please run 'summarizer.py' first to generate this file.")
#         return None
#     except json.JSONDecodeError:
#         logging.error(f"FATAL: Could not parse JSON from '{filename}'.")
#         return None

# def generate_vector(summary_text, model, tokenizer):
#     """
#     Task 4.1: Generates an initial vector from a text summary.
#     """
#     logging.debug(f"Generating vector for summary: '{summary_text[:60]}...'")
#     try:
#         inputs = tokenizer(
#             summary_text,
#             return_tensors="pt",
#             truncation=True,
#             padding=True
#         )
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         # Get the "pooled" output (a single 768-dim vector for the summary)
#         initial_vector = outputs.pooler_output[0]
        
#         # Convert to a NumPy array for FAISS and privacy noise
#         return initial_vector.numpy()
        
#     except Exception as e:
#         logging.error(f"Error generating vector for '{summary_text}': {e}")
#         return None

# def main():
#     """
#     Runs the full Phase 4 pipeline:
#     1. Loads summaries
#     2. Loads AI model
#     3. Generates vectors (Task 4.1)
#     4. Applies noise (Task 4.2)
#     5. Builds and saves the FAISS vector database (Task 4.3)
#     """
#     logging.info("--- [ Phase 4: Building the 'Semantic Brain' ] ---")

#     # 1. Load the AI "brain" (from Task 2.1)
#     model, tokenizer = load_ai_model()
#     if not model or not tokenizer:
#         logging.fatal("Failed to load AI model. Exiting.")
#         sys.exit(1)
        
#     # 2. Load the Anonymized Summaries (Output from Phase 3)
#     summaries = load_summaries(INPUT_SUMMARIES_FILE)
#     if not summaries:
#         logging.fatal("Failed to load summaries file. Exiting.")
#         sys.exit(1)

#     # 3. Run the full Phase 4 pipeline
#     logging.info("Starting vector generation and privacy application...")
    
#     all_vectors = []         # This will hold the final NumPy vectors
#     metadata_mapping = {}    # This will map an index ID to its info
#     vector_index_id = 0
    
#     for file_path, nodes in summaries.items():
#         logging.info(f"Processing {len(nodes)} nodes for: {file_path}")
        
#         for node in nodes:
#             summary = node.get('summary', '')
#             node_name = node.get('name', 'UnnamedNode')
            
#             if not summary:
#                 logging.warning(f"Skipping {node_name}, no summary found.")
#                 continue

#             # --- Task 4.1: Generate Initial Vector ---
#             initial_vector = generate_vector(summary, model, tokenizer)
#             if initial_vector is None:
#                 logging.warning(f"Failed to generate vector for {node_name}, skipping.")
#                 continue

#             # --- Task 4.2: Apply Differential Privacy "Noise" ---
#             noisy_vector = add_laplacian_noise(initial_vector)
            
#             # --- Task 4.3: Store Vector and Mapping ---
#             all_vectors.append(noisy_vector)
            
#             # Store the human-readable info for this vector
#             metadata_mapping[vector_index_id] = {
#                 'file_path': file_path,
#                 'node_type': node.get('type', 'Unknown'),
#                 'node_name': node_name
#             }
#             vector_index_id += 1

#     logging.info(f"\nSuccessfully generated {len(all_vectors)} privacy-preserving vectors.")

#     # 4. Build and Save the FAISS Index (Task 4.3)
#     if not all_vectors:
#         logging.warning("No vectors were generated. Exiting.")
#         return

#     try:
#         # Convert our list of vectors into a single 2D NumPy array
#         # FAISS requires a float32 array
#         vectors_np = np.array(all_vectors).astype('float32')
        
#         # Normalize the vectors (good practice for L2/cosine similarity)
#         faiss.normalize_L2(vectors_np)

#         # Create a new, empty FAISS index
#         # IndexFlatL2 = a simple, exact-search index using L2 distance
#         index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        
#         # Add our vectors to the index
#         index.add(vectors_np)
        
#         logging.info(f"Successfully built FAISS index with {index.ntotal} vectors.")

#         # Save the index to disk
#         faiss.write_index(index, OUTPUT_INDEX_FILE)
#         logging.info(f"Saved FAISS index to: {OUTPUT_INDEX_FILE}")
        
#         # Save the metadata mapping
#         with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
#             json.dump(metadata_mapping, f, indent=2)
#         logging.info(f"Saved metadata mapping to: {OUTPUT_MAPPING_FILE}")
        
#         logging.info("\n--- [ Phase 4 Complete ] ---")
#         logging.info("The 'Semantic Brain' is built and ready for queries.")

#     except Exception as e:
#         logging.error(f"FATAL: Could not build or save FAISS index: {e}")


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
    import faiss
    import torch
except ImportError:
    logging.basicConfig()
    logging.error("="*60)
    logging.error("FATAL: 'numpy' or 'faiss-cpu' or 'torch' library not found.")
    logging.error("Please install them by running:")
    logging.error("pip install numpy faiss-cpu torch")
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
    format='%(asctime)s - %(levelname)s - [BRAIN_BUILDER] - %(message)s'
)
INPUT_SUMMARIES_FILE = 'anonymized_summaries.json'
OUTPUT_FAISS_INDEX = 'semantic_brain.faiss'
OUTPUT_MAPPING_FILE = 'semantic_brain_mapping.json'
VECTOR_DIMENSION = 768 # CodeBERT (base) outputs 768-dimension vectors
# --- End Configuration ---


def generate_vector_embedding(text, model, tokenizer):
    """
    Task 4.1: Generate Initial Vector
    
    Converts a single text summary into its initial CodeBERT vector.
    """
    try:
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        
        # We use pooler_output as the semantic vector for the [CLS] token
        initial_vector = outputs.pooler_output[0].numpy()
        return initial_vector
        
    except Exception as e:
        logging.error(f"Failed to generate vector for text: '{text}'. Error: {e}")
        return None

def main():
    """
    Main execution function for Phase 4.
    """
    logging.info("--- [ Phase 4: Building the 'Semantic Brain' ] ---")

    # 1. Load the AI "brain" (CodeBERT)
    model, tokenizer = load_ai_model()
    if not model or not tokenizer:
        logging.fatal("Failed to load CodeBERT model. Exiting.")
        sys.exit(1)

    # 2. Load the "Output 1" summaries
    logging.info(f"Loading summaries from '{INPUT_SUMMARIES_FILE}'...")
    try:
        with open(INPUT_SUMMARIES_FILE, 'r', encoding='utf-8') as f:
            summaries_registry = json.load(f)
    except FileNotFoundError:
        logging.error(f"FATAL: Input file '{INPUT_SUMMARIES_FILE}' not found.")
        logging.error("Please run a summarizer script first.")
        sys.exit(1)

    # 3. Create FAISS index (Task 4.3)
    # We use IndexFlatL2, a simple (but fast) index for L2 (Euclidean) distance
    index = faiss.IndexFlatL2(VECTOR_DIMENSION)
    
    vector_data = []
    vector_mapping = {} # This will map the vector's ID (e.g., 0, 1, 2) back to our code
    vector_id_counter = 0

    logging.info("Starting vector generation...")
    
    # 4. Loop through all summaries and generate vectors
    for file_path, nodes in summaries_registry.items():
        for node in nodes:
            summary = node.get('summary')
            if not summary:
                logging.warning(f"Skipping node '{node['name']}' (no summary).")
                continue
            
            # --- Task 4.1: Generate Initial Vector ---
            initial_vector = generate_vector_embedding(summary, model, tokenizer)
            if initial_vector is None:
                continue
            
            # --- Task 4.2: Apply Differential Privacy "Noise" ---
            noisy_vector = add_laplacian_noise(initial_vector, SENSITIVITY, EPSILON)
            
            # Add the final, noisy vector to our list
            vector_data.append(noisy_vector)
            
            # Create the map entry for this vector
            vector_mapping[vector_id_counter] = {
                "file_path": file_path,
                "node_type": node['type'],
                "node_name": node['name']
            }
            vector_id_counter += 1
            
            logging.debug(f"Generated vector for: {node['name']}")

    if not vector_data:
        logging.error("No vectors were generated. Is the summaries file empty?")
        sys.exit(1)
        
    # 5. Add all vectors to the FAISS index
    logging.info(f"Adding {len(vector_data)} noisy vectors to the FAISS index...")
    # FAISS requires a 2D NumPy array
    vector_data_np = np.array(vector_data).astype('float32')
    index.add(vector_data_np)

    # 6. Save the final "Semantic Brain" (Task 4.3)
    logging.info(f"Saving FAISS index to '{OUTPUT_FAISS_INDEX}'...")
    faiss.write_index(index, OUTPUT_FAISS_INDEX)
    
    logging.info(f"Saving vector mapping to '{OUTPUT_MAPPING_FILE}'...")
    with open(OUTPUT_MAPPING_FILE, 'w', encoding='utf-8') as f:
        json.dump(vector_mapping, f, indent=4)
        
    logging.info("--- [ Phase 4 Complete ] ---")
    logging.info(f"Successfully created '{OUTPUT_FAISS_INDEX}'.")
    logging.info("The 'Semantic Brain' is built and ready for validation.")

if __name__ == "__main__":
    main()

