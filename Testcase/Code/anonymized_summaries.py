# import logging
# import sys
# import json

# # Import the anonymizer from Phase 2
# try:
#     from anonymizer import anonymize_code_chunk
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'anonymizer.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [SUMMARIZER] - %(message)s'
# )

# # Define our input/output files for this Phase
# INPUT_FILE = "hierarchical_nodes.json"
# OUTPUT_FILE = "anonymized_summaries.json"
# # --- End Configuration ---


# # --- [ Task 3.1: Semantic Summarization (Simulated) ] ---
# def generate_semantic_summary(node: dict, raw_code_chunk: str) -> str:
#     """
#     (SIMULATED) Generates a semantic summary for a code chunk.
    
#     *** EDUCATIONAL NOTE ***
#     This is a placeholder! In a full-scale production system, this
#     function would pass the 'raw_code_chunk' to a generative LLM
#     (like a fine-tuned T5, BART, or a model like GPT-4) to create
#     a high-quality, natural-language summary.
    
#     Since our CodeBERT model (from Task 2.1) is an *encoder* (creates
#     vectors) and not a *generator* (writes text), we simulate this
#     step with generalized rules.
    
#     This simulation is "generalized" and not hard-coded
#     to specific names, but still injects "dirty" data
#     for Task 3.2 (Anonymization) to clean.
#     """
#     node_type = node['type']
#     node_name = node['name']

#     # --- [ Generalized, Fake Summaries for Demonstration ] ---
#     if node_type == 'ClassDef':
#         # e.g., "A class definition for 'PaymentService'. It likely manages Payment logic
#         # for [PROPRIETARY_NAME] and handles currencies like USD, EUR."
#         class_intent = node_name.replace('Service', '').replace('Controller', '').replace('Manager', '')
#         return (f"A class definition for '{node_name}'. It likely manages {class_intent} logic "
#                 f"for [PROPRIETARY_NAME] and handles currencies like USD, EUR.")
    
#     if node_type == 'FunctionDef' or node_type == 'AsyncFunctionDef':
#         # Try to guess intent from name
#         intent = "data"
#         if '_' in node_name:
#             intent = node_name.split('_')[-1]
        
#         # Inject different kinds of "dirty" data based on name
#         if 'db' in node_name or 'database' in node_name or 'sql' in node_name:
#              return (f"A function '{node_name}' that connects to the database "
#                      f"using 'postgres://user:pass@host/db' to process {intent}.")
        
#         if 'api' in node_name or 'key' in node_name or 'payment' in node_name:
#             return (f"A function '{node_name}' that handles {intent}. It may use a "
#                     f"secret key like sk_live_abc123... or API_KEY = 'abc123xyz'")

#         # Generic fallback for other functions
#         return (f"A function '{node_name}' that processes {intent}. "
#                 f"It might use an internal key like 'PROJECT_PHOENIX_KEY'.")

#     return f"A general code chunk for {node_name}."
#     # --- [ End Generalized Summaries ] ---


# # --- [ Task 3.2: Privacy-Preserving Anonymization ] ---
# def anonymize_summary(summary_text: str) -> str:
#     """
#     Applies the rules from anonymizer.py to a *text summary*.
#     This is the core of Task 3.2.
#     """
#     # We can re-use the same anonymizer because it's just regex!
#     return anonymize_code_chunk(summary_text)


# # --- [ Task 3.3: Format and Store Output 1 ] ---
# def process_and_summarize(node_registry: dict) -> dict:
#     """
#     Orchestrates all of Phase 3.
#     Iterates the registry, generates a summary, anonymizes it,
#     and builds the final JSON structure for Output 1.
#     """
#     logging.info("--- [ Phase 3: Generating Anonymized Summaries ] ---")
    
#     # This will be our final JSON output
#     # Format: { "filepath": [list of summary nodes] }
#     summaries_registry = {}

#     for file_path, nodes in node_registry.items():
#         summaries_registry[file_path] = []
#         logging.info(f"Processing file: {file_path}...")
        
#         for node in nodes:
#             # --- [ EDIT: Made this more robust ] ---
#             # Get the raw, anonymized code (from node_extractor.py)
#             # We don't use it in the simulation, but a real LLM would.
#             # We use .get() to be robust against older JSON files.
#             raw_code_chunk = node.get('anonymized_source_code', '')
            
#             if not raw_code_chunk:
#                 # Log a warning if the key was missing
#                 logging.warning(f"  - Node '{node['name']}' is missing 'anonymized_source_code' key.")
#                 logging.warning("    (This is OK, but please re-run 'node_extractor.py' to get the latest JSON.)")
#             # --- [ End Edit ] ---

#             # --- Task 3.1 ---
#             # Generate the "semantic" summary (e.g., "...handles USD...")
#             raw_summary = generate_semantic_summary(node, raw_code_chunk)
            
#             # --- Task 3.2 ---
#             # Anonymize the summary (e.g., "...handles [CURRENCY]...")
#             final_summary = anonymize_summary(raw_summary)
            
#             # --- Task 3.3 ---
#             # Format the final node for our JSON output
#             # We only need the type, name, and summary for the KG
#             summary_node = {
#                 "type": node['type'],
#                 "name": node['name'],
#                 "summary": final_summary
#             }
#             summaries_registry[file_path].append(summary_node)
            
#             # Set log level to DEBUG to see this
#             logging.debug(f"  - Generated summary for: {node['name']}")
#             logging.debug(f"    Raw:      '{raw_summary}'")
#             logging.debug(f"    Anonymized: '{final_summary}'")

#     logging.info("Successfully generated all anonymized summaries.")
#     return summaries_registry


# def main():
#     """
#     Main execution function for Phase 3.
#     """
#     # 1. Load the input from Phase 1
#     try:
#         logging.info(f"Loading hierarchical nodes from '{INPUT_FILE}'...")
#         with open(INPUT_FILE, 'r', encoding='utf-8') as f:
#             node_registry = json.load(f)
#         logging.info(f"Found {len(node_registry)} files to process.")
#     except FileNotFoundError:
#         logging.error(f"FATAL: Input file '{INPUT_FILE}' not found.")
#         logging.error("Please run 'node_extractor.py' first to generate this file.")
#         sys.exit(1)
#     except json.JSONDecodeError:
#         logging.error(f"FATAL: Could not parse '{INPUT_FILE}'. Is it corrupted?")
#         sys.exit(1)

#     # 2. Run the full Phase 3 pipeline
#     summaries_data = process_and_summarize(node_registry)

#     # 3. Export the final output (Task 3.3)
#     try:
#         logging.info(f"Exporting anonymized summaries to '{OUTPUT_FILE}'...")
#         with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
#             json.dump(summaries_data, f, indent=2)
#         logging.info(f"--- [ Phase 3 Complete ] ---")
#         logging.info(f"Successfully created '{OUTPUT_FILE}'.")
#         logging.info("This file is 'Output 1' and is ready for Phase 5 (Knowledge Graph).")
#     except IOError as e:
#         logging.error(f"Failed to write output file: {e}")
#     except TypeError as e:
#         logging.error(f"Failed to serialize final JSON: {e}")

# if __name__ == "__main__":
#     main()



import logging
import sys
import json

# Import the anonymizer from Phase 2
try:
    from anonymizer import anonymize_code_chunk
except ImportError:
    logging.basicConfig() # Ensure logging is configured for the error
    logging.error("="*60)
    logging.error("FATAL: 'anonymizer.py' not found.")
    logging.error("Please make sure all .py scripts are in the same directory.")
    logging.error("="*60)
    sys.exit(1) # Exit if the dependency is missing

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [SUMMARIZER] - %(message)s'
)

# Define our input/output files for this Phase
INPUT_FILE = "hierarchical_nodes.json"
OUTPUT_FILE = "anonymized_summaries.json"
# --- End Configuration ---


# --- [ Task 3.1: Semantic Summarization (Simulated) ] ---
def generate_semantic_summary(node: dict, raw_code_chunk: str) -> str:
    """
    (SIMULATED) Generates a semantic summary for a code chunk.
    
    *** [ EDIT: This is now a "smarter" simulation ] ***
    
    This simulation now uses the 'raw_code_chunk' to make more
    intelligent guesses, which is much better than just using the name.
    
    A real production system would replace this function with a call
    to a generative LLM (like T5, BART, or GPT).
    """
    node_type = node['type']
    node_name = node['name']
    
    # --- [ NEW: Smarter Simulation Logic ] ---
    
    # 1. Get some basic stats from the raw code
    lines_of_code = len(raw_code_chunk.split('\n'))
    has_imports = "import " in raw_code_chunk
    has_return = "return " in raw_code_chunk
    
    # 2. Create a more descriptive summary
    summary_parts = []

    if node_type == 'ClassDef':
        class_intent = node_name.replace('Service', '').replace('Controller', '').replace('Manager', '')
        summary_parts.append(f"A class definition for '{node_name}' ({lines_of_code} lines).")
        summary_parts.append(f"It appears to manage logic related to '{class_intent}'.")
    
    elif node_type in ('FunctionDef', 'AsyncFunctionDef'):
        intent = "data"
        if '_' in node_name:
            intent = node_name.split('_')[-1]
        
        summary_parts.append(f"A function '{node_name}' ({lines_of_code} lines).")
        summary_parts.append(f"It seems to process {intent}.")

    # Add more details based on code content
    if has_imports:
        summary_parts.append("It handles its own imports.")
    if has_return:
        summary_parts.append("It returns a value.")
        
    # Inject "dirty" data for anonymization to clean
    if 'db' in node_name or 'sql' in node_name:
         summary_parts.append("It may connect to a DB like 'postgres://user:pass@host/db'.")
    elif 'api' in node_name or 'payment' in node_name:
        summary_parts.append("It may use a secret key like sk_live_abc123... and handle USD.")
    else:
        summary_parts.append("It might use an internal key like 'PROJECT_PHOENIX_KEY'.")

    return " ".join(summary_parts)
    # --- [ End Smarter Simulation Logic ] ---


# --- [ Task 3.2: Privacy-Preserving Anonymization ] ---
def anonymize_summary(summary_text: str) -> str:
    """
    Applies the rules from anonymizer.py to a *text summary*.
    This is the core of Task 3.2.
    """
    # We can re-use the same anonymizer because it's just regex!
    return anonymize_code_chunk(summary_text)


# --- [ Task 3.3: Format and Store Output 1 ] ---
def process_and_summarize(node_registry: dict) -> dict:
    """
    Orchestrates all of Phase 3.
    Iterates the registry, generates a summary, anonymizes it,
    and builds the final JSON structure for Output 1.
    """
    logging.info("--- [ Phase 3: Generating Anonymized Summaries ] ---")
    
    # This will be our final JSON output
    # Format: { "filepath": [list of summary nodes] }
    summaries_registry = {}

    for file_path, nodes in node_registry.items():
        summaries_registry[file_path] = []
        logging.info(f"Processing file: {file_path}...")
        
        for node in nodes:
            # --- [ EDIT: Made this more robust ] ---
            # Get the raw, anonymized code (from node_extractor.py)
            # We don't use it in the simulation, but a real LLM would.
            # We use .get() to be robust against older JSON files.
            raw_code_chunk = node.get('anonymized_source_code', '')
            
            if not raw_code_chunk:
                # Log a warning if the key was missing
                logging.warning(f"  - Node '{node['name']}' is missing 'anonymized_source_code' key.")
                logging.warning("    (This is OK, but please re-run 'node_extractor.py' to get the latest JSON.)")
            # --- [ End Edit ] ---

            # --- Task 3.1 ---
            # Generate the "semantic" summary (e.g., "...handles USD...")
            raw_summary = generate_semantic_summary(node, raw_code_chunk)
            
            # --- Task 3.2 ---
            # Anonymize the summary (e.g., "...handles [CURRENCY]...")
            final_summary = anonymize_summary(raw_summary)
            
            # --- Task 3.3 ---
            # Format the final node for our JSON output
            # We only need the type, name, and summary for the KG
            summary_node = {
                "type": node['type'],
                "name": node['name'],
                "summary": final_summary
            }
            summaries_registry[file_path].append(summary_node)
            
            # Set log level to DEBUG to see this
            logging.debug(f"  - Generated summary for: {node['name']}")
            logging.debug(f"    Raw:      '{raw_summary}'")
            logging.debug(f"    Anonymized: '{final_summary}'")

    logging.info("Successfully generated all anonymized summaries.")
    return summaries_registry


def main():
    """
    Main execution function for Phase 3.
    """
    # 1. Load the input from Phase 1
    try:
        logging.info(f"Loading hierarchical nodes from '{INPUT_FILE}'...")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            node_registry = json.load(f)
        logging.info(f"Found {len(node_registry)} files to process.")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file '{INPUT_FILE}' not found.")
        logging.error("Please run 'node_extractor.py' first to generate this file.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not parse '{INPUT_FILE}'. Is it corrupted?")
        sys.exit(1)

    # 2. Run the full Phase 3 pipeline
    summaries_data = process_and_summarize(node_registry)

    # 3. Export the final output (Task 3.3)
    try:
        logging.info(f"Exporting anonymized summaries to '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(summaries_data, f, indent=2)
        logging.info(f"--- [ Phase 3 Complete ] ---")
        logging.info(f"Successfully created '{OUTPUT_FILE}'.")
        logging.info("This file is 'Output 1' and is ready for Phase 5 (Knowledge Graph).")
    except IOError as e:
        logging.error(f"Failed to write output file: {e}")
    except TypeError as e:
        logging.error(f"Failed to serialize final JSON: {e}")

if __name__ == "__main__":
    main()

