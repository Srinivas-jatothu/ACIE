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
    
#     *** [ EDIT: This is now a "smarter" simulation ] ***
    
#     This simulation now uses the 'raw_code_chunk' to make more
#     intelligent guesses, which is much better than just using the name.
    
#     A real production system would replace this function with a call
#     to a generative LLM (like T5, BART, or GPT).
#     """
#     node_type = node['type']
#     node_name = node['name']
    
#     # --- [ NEW: Smarter Simulation Logic ] ---
    
#     # 1. Get some basic stats from the raw code
#     lines_of_code = len(raw_code_chunk.split('\n'))
#     has_imports = "import " in raw_code_chunk
#     has_return = "return " in raw_code_chunk
    
#     # 2. Create a more descriptive summary
#     summary_parts = []

#     if node_type == 'ClassDef':
#         class_intent = node_name.replace('Service', '').replace('Controller', '').replace('Manager', '')
#         summary_parts.append(f"A class definition for '{node_name}' ({lines_of_code} lines).")
#         summary_parts.append(f"It appears to manage logic related to '{class_intent}'.")
    
#     elif node_type in ('FunctionDef', 'AsyncFunctionDef'):
#         intent = "data"
#         if '_' in node_name:
#             intent = node_name.split('_')[-1]
        
#         summary_parts.append(f"A function '{node_name}' ({lines_of_code} lines).")
#         summary_parts.append(f"It seems to process {intent}.")

#     # Add more details based on code content
#     if has_imports:
#         summary_parts.append("It handles its own imports.")
#     if has_return:
#         summary_parts.append("It returns a value.")
        
#     # Inject "dirty" data for anonymization to clean
#     if 'db' in node_name or 'sql' in node_name:
#          summary_parts.append("It may connect to a DB like 'postgres://user:pass@host/db'.")
#     elif 'api' in node_name or 'payment' in node_name:
#         summary_parts.append("It may use a secret key like sk_live_abc123... and handle USD.")
#     else:
#         summary_parts.append("It might use an internal key like 'PROJECT_PHOENIX_KEY'.")

#     return " ".join(summary_parts)
#     # --- [ End Smarter Simulation Logic ] ---


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
import os
import asyncio  # Import for asynchronous API calls
from aiohttp import ClientSession, ClientError, TCPConnector # For async HTTP

# --- Task 2.2 Import: The "Anonymizer" ---
# We still need this to "scrub" the summaries after the AI generates them.
try:
    from anonymizer import anonymize_code_chunk
except ImportError:
    logging.basicConfig()
    logging.error("=" * 60)
    logging.error("FATAL: 'anonymizer.py' not found.")
    logging.error("Please make sure 'anonymizer.py' is in the same directory.")
    logging.error("=" * 60)
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [SMART_SUMMARIZER] - %(message)s'
)
INPUT_NODES_FILE = 'hierarchical_nodes.json'
OUTPUT_SUMMARIES_FILE = 'anonymized_summaries.json'

# --- Gemini API Configuration ---
# We leave the API key blank, as it will be provided by the environment.
API_KEY = "AIzaSyDbHfMRiMtRoG9xHGRHOCdKCkr8cBVjrOo"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={API_KEY}"
SYSTEM_PROMPT = (
    "You are an expert principal engineer. Your task is to write a single, "
    "concise, one-sentence summary for a given chunk of code. Focus on the "
    "*business purpose* or *intent* of the code, not just what it *does*. "
    "For example, 'Handles payment processing' is better than 'Calls the Stripe API'. "
    "Do not mention the programming language. Do not just list the function's arguments. "
    "BAD: 'A function that takes a cart and returns a boolean.' "
    "GOOD: 'Validates a shopping cart to ensure item counts are within limits.'"
)
# --- End Configuration ---


async def fetch_with_backoff(session, url, payload, retries=5, initial_delay=1):
    """
    Performs an async POST request with exponential backoff.
    This makes our script robust to temporary network or API errors.
    """
    delay = initial_delay
    for i in range(retries):
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    # Extract the text from the Gemini API response
                    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
                    if text:
                        return text
                    else:
                        logging.warning(f"API call successful but no text in response: {result}")
                        return "Error: Could not parse AI response."
                else:
                    logging.warning(
                        f"API call failed with status {response.status}: {await response.text()}. "
                        f"Retrying in {delay}s... (Attempt {i + 1}/{retries})"
                    )

        except ClientError as e:
            logging.warning(
                f"Network error during API call: {e}. "
                f"Retrying in {delay}s... (Attempt {i + 1}/{retries})"
            )
        except asyncio.TimeoutError:
            logging.warning(
                f"API call timed out. Retrying in {delay}s... (Attempt {i + 1}/{retries})"
            )
            
        await asyncio.sleep(delay)
        delay *= 2  # Exponentially increase the delay

    logging.error(f"Failed to fetch summary after {retries} retries.")
    return "Error: Failed to generate summary after multiple retries."


async def generate_semantic_summary(session, anonymized_code, node_name):
    """
    Task 3.1: Generate Semantic Summarization (The "Smart" Way)
    
    This function sends the (already safe) anonymized code to the 
    Gemini API to get a high-quality, human-readable summary.
    """
    logging.debug(f"Generating summary for: {node_name}")

    # 1. Create the prompt for the AI
    user_query = (
        f"Summarize this code chunk, which is named '{node_name}':\n\n"
        f"```\n{anonymized_code}\n```"
    )

    # 2. Construct the API payload
    payload = {
        "contents": [{
            "parts": [{"text": user_query}]
        }],
        "systemInstruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 100,
        }
    }
    
    # 3. Call the API using our robust fetcher
    # This (Task 3.1) generates the "dirty" summary (e.g., "Handles USD...")
    dirty_summary = await fetch_with_backoff(session, API_URL, payload)
    
    # 4. Task 3.2: Anonymize the *summary itself*
    # This scrubs the AI's output, just in case it repeated a secret
    # (e.g., "Handles USD..." -> "Handles [CURRENCY]...")
    clean_summary = anonymize_code_chunk(dirty_summary)
    
    return clean_summary, node_name


async def process_and_summarize(node_registry):
    """
    Task 3.3: Format and Store Output 1
    
    This function orchestrates the entire summarization process,
    using asyncio.gather to run all API calls concurrently.
    """
    summaries_data = {}
    tasks = []
    node_references = [] # To map results back to the JSON structure

    # Use a single, shared ClientSession for all requests
    # Limit to 10 concurrent connections to avoid rate-limiting
    async with ClientSession(connector=TCPConnector(limit=10)) as session:
        
        # --- Stage 1: Create all API call "tasks" ---
        logging.info("Preparing to summarize code chunks...")
        for file_path, nodes in node_registry.items():
            summaries_data[file_path] = []
            for node_info in nodes:
                node_name = node_info.get("name", "Unnamed Node")
                
                # We use the anonymized source code we created in Phase 1
                anonymized_code = node_info.get("anonymized_source_code")

                if not anonymized_code:
                    logging.warning(f"Skipping node '{node_name}' (no code found).")
                    continue
                
                # Create a new dict for our final output
                new_node_info = {
                    "type": node_info.get("type"),
                    "name": node_name,
                    "summary": "" # Will be filled in Stage 3
                }
                summaries_data[file_path].append(new_node_info)
                
                # Add the API call to our list of tasks
                tasks.append(
                    generate_semantic_summary(session, anonymized_code, node_name)
                )
                # Keep a reference to the dict we just created
                node_references.append(new_node_info)

        if not tasks:
            logging.warning("No code chunks found to summarize.")
            return {}

        # --- Stage 2: Run all tasks concurrently ---
        logging.info(f"Sending {len(tasks)} code chunks to AI for summarization (in parallel)...")
        # asyncio.gather runs all "tasks" at the same time and waits for them all to finish.
        # This is *much* faster than doing them one by one.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        logging.info("All AI summaries have been received.")

        # --- Stage 3: Populate results back into our JSON structure ---
        for i, result in enumerate(results):
            # Find the original dict we created for this task
            node_to_update = node_references[i]
            
            if isinstance(result, Exception):
                logging.error(f"Failed to generate summary for '{node_to_update['name']}': {result}")
                node_to_update['summary'] = "Error: Task failed."
            else:
                # Result is a tuple: (clean_summary, node_name)
                clean_summary, node_name = result
                logging.info(f"Summary for '{node_name}': {clean_summary}")
                node_to_update['summary'] = clean_summary
                
    return summaries_data


async def main():
    """
    Asynchronous main function to run the pipeline.
    """
    logging.info("--- [ Phase 3: Smart Summarization ] ---")

    # 1. Load the hierarchical nodes
    logging.info(f"Loading hierarchical nodes from '{INPUT_NODES_FILE}'...")
    try:
        with open(INPUT_NODES_FILE, 'r', encoding='utf-8') as f:
            node_registry = json.load(f)
        logging.info(f"Found {len(node_registry)} files to process.")
    except FileNotFoundError:
        logging.error(f"FATAL: Input file '{INPUT_NODES_FILE}' not found.")
        logging.error("Please run 'node_extractor.py' first to generate this file.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"FATAL: Could not parse JSON from '{INPUT_NODES_FILE}'.")
        sys.exit(1)

    # 2. Run the summarization and anonymization pipeline
    summaries_data = await process_and_summarize(node_registry)

    # 3. Export the final, "Output 1" JSON
    if summaries_data:
        logging.info(f"Exporting anonymized summaries to '{OUTPUT_SUMMARIES_FILE}'...")
        try:
            with open(OUTPUT_SUMMARIES_FILE, 'w', encoding='utf-8') as f:
                json.dump(summaries_data, f, indent=4)
            logging.info("--- [ Phase 3 Complete ] ---")
            logging.info(f"Successfully created '{OUTPUT_SUMMARIES_FILE}'.")
            logging.info("This file is 'Output 1' and is ready for Phase 5 (Knowledge Graph).")
        except (IOError, TypeError) as e:
            logging.error(f"Failed to write output JSON: {e}")

# This is the standard way to run an "async main" function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.fatal(f"An unhandled error occurred: {e}")
        sys.exit(1)
