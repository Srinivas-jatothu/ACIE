# import os
# import logging
# import ast  # Python's built-in Abstract Syntax Tree module

# # --- Task 1.1 Import ---
# # This script assumes 'source_collector.py' is in the same directory
# # We import the function and settings from your first file.
# try:
#     from source_collector import scan_codebase, SUPPORTED_EXTENSIONS
# except ImportError:
#     logging.error("FATAL: Could not import 'source_collector.py'.")
#     logging.error("Please make sure 'source_collector.py' is in the same folder.")
#     exit()
# # -----------------------


# # --- Configuration ---
# # Set up logging for this script
# logging.basicConfig(
#     level=logging.INFO, # INFO level is good, DEBUG is too noisy for ASTs
#     format='%(asctime)s - %(levelname)s - [AST_PARSER] - %(message)s'
# )
# # --- End Configuration ---


# # --- Language-Specific Parsers ---

# def parse_python_ast(file_content, file_path):
#     """
#     Parses raw Python code into an AST object.
    
#     Input: Raw text content of a .py file.
#     Output: An in-memory AST object.
#     """
#     try:
#         ast_tree = ast.parse(file_content)
#         logging.info(f"Successfully parsed Python AST for: {file_path}")
#         return ast_tree
#     except SyntaxError as e:
#         logging.error(f"Syntax error in {file_path}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Failed to parse {file_path}: {e}")
#         return None

# def parse_java_ast(file_content, file_path):
#     """
#     Placeholder for a Java AST parser.
#     You would need a library like 'plyj' or 'javalang'.
#     """
#     logging.warning(f"Java AST parser not implemented. Skipping: {file_path}")
#     # Example (if you had javalang installed):
#     # try:
#     #   return javalang.parse.parse(file_content)
#     # except:
#     #   return None
#     return None

# def parse_cpp_ast(file_content, file_path):
#     """
#     Placeholder for a C++ AST parser.
#     You would need a library like 'clang'.
#     """
#     logging.warning(f"C++ AST parser not implemented. Skipping: {file_path}")
#     return None

# # --- Add other language parsers here (e.g., for TS, JS, Go) ---

# # This map links file extensions to their specific parser function
# PARSER_MAP = {
#     '.py': parse_python_ast,
#     '.java': parse_java_ast,
#     '.cpp': parse_cpp_ast,
#     '.c': parse_cpp_ast, # C and C++ can often use the same frontend
#     '.h': parse_cpp_ast,
#     # Add other extensions as you implement parsers
#     # '.js': parse_javascript_ast,
#     # '.ts': parse_typescript_ast,
# }


# def parse_file_to_ast(file_path):
#     """
#     Reads a file and routes it to the correct language-specific AST parser.

#     Input: A file path (str).
#     Output: An in-memory AST object, or None if parsing fails or is unsupported.
#     """
#     file_ext = os.path.splitext(file_path)[1].lower()
    
#     # Find the correct parser function from our map
#     parser_func = PARSER_MAP.get(file_ext)
    
#     if not parser_func:
#         logging.debug(f"No AST parser defined for extension {file_ext}. Skipping.")
#         return None

#     # Try to read the file content
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except (IOError, UnicodeDecodeError) as e:
#         logging.error(f"Could not read file {file_path}: {e}")
#         return None
        
#     # Call the specific parser with the file content
#     return parser_func(content, file_path)


# if __name__ == "__main__":
#     # --- Main Execution ---
    
#     # Set the target directory (same as in source_collector.py)
#     target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"
    
#     # --- Task 1.1: Run the collector ---
#     logging.info("--- [ Task 1.1: Collecting Source Files ] ---")
#     file_registry = scan_codebase(target_project_path)
    
#     if not file_registry:
#         logging.info("No files found. Exiting.")
#         exit()
        
#     logging.info(f"Found {len(file_registry)} files to parse.")

#     # --- Task 1.2: Run the parser ---
#     logging.info("\n--- [ Task 1.2: Parsing ASTs ] ---")
    
#     # This registry will hold our in-memory AST objects
#     # Key: file_path, Value: ast_object
#     ast_registry = {}

#     for file_record in file_registry:
#         file_path = file_record['path']
#         ast_obj = parse_file_to_ast(file_path)
        
#         if ast_obj:
#             ast_registry[file_path] = ast_obj

#     # --- Summary & Debugging ---
#     logging.info("\n--- [ AST Parsing Summary ] ---")
#     logging.info(f"Total files scanned: {len(file_registry)}")
#     logging.info(f"Successfully parsed ASTs: {len(ast_registry)}")
#     logging.info("Files parsed:")
#     for path in ast_registry.keys():
#         logging.info(f"  - {path}")
        
#     unparsed_count = len(file_registry) - len(ast_registry)
#     if unparsed_count > 0:
#         logging.info(f"Skipped/Failed {unparsed_count} files (see warnings above).")

#     # --- Example AST "Debug" ---
#     # Let's show a small part of the first Python AST we parsed
#     if ast_registry:
#         try:
#             # Find the first Python AST
#             py_ast_example = None
#             first_path = ""
#             for path, ast_obj in ast_registry.items():
#                 if path.endswith('.py'):
#                     py_ast_example = ast_obj
#                     first_path = path
#                     break
            
#             if py_ast_example:
#                 logging.info(f"\n--- [ Example AST 'Debug' for: {first_path} ] ---")
#                 logging.info("Dumping top-level nodes (Classes and Functions):")
                
#                 # We can walk the top level of the AST 'body'
#                 for top_level_node in py_ast_example.body:
#                     if isinstance(top_level_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                         logging.info(f"  > Found Function: {top_level_node.name}")
#                     elif isinstance(top_level_node, ast.ClassDef):
#                         logging.info(f"  > Found Class: {top_level_node.name}")
#                     elif isinstance(top_level_node, ast.Import):
#                         logging.info(f"  > Found Import")
#                     elif isinstance(top_level_node, ast.ImportFrom):
#                         logging.info(f"  > Found ImportFrom: {top_level_node.module}")
                
#                 # For a full, *very* verbose tree, you can use ast.dump()
#                 # logging.debug(ast.dump(py_ast_example, indent=4))

#         except Exception as e:
#             logging.error(f"Error during AST example dump: {e}")




# import os
# import logging
# import ast  # Python's built-in Abstract Syntax Tree module

# # --- Task 1.1 Import ---
# # This script assumes 'source_collector.py' is in the same directory
# # We import the function and settings from your first file.
# try:
#     from source_collector import scan_codebase, SUPPORTED_EXTENSIONS
# except ImportError:
#     logging.error("FATAL: Could not import 'source_collector.py'.")
#     logging.error("Please make sure 'source_collector.py' is in the same folder.")
#     exit()
# # -----------------------


# # --- Configuration ---
# # Set up logging for this script.
# # NOTE: The logging config from the *imported* 'source_collector'
# # will likely take precedence. This is fine.
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [AST_PARSER] - %(message)s'
# )
# # --- End Configuration ---


# # --- Language-Specific Parsers ---

# def parse_python_ast(file_content, file_path):
#     """
#     Parses raw Python code into an AST object.
    
#     Input: Raw text content of a .py file.
#     Output: An in-memory AST object.
#     """
#     try:
#         ast_tree = ast.parse(file_content)
#         logging.info(f"Successfully parsed Python AST for: {file_path}")
#         return ast_tree
#     except SyntaxError as e:
#         logging.error(f"Syntax error in {file_path}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Failed to parse {file_path}: {e}")
#         return None

# def parse_java_ast(file_content, file_path):
#     """
#     Placeholder for a Java AST parser.
#     You would need a library like 'plyj' or 'javalang'.
#     """
#     logging.warning(f"Java AST parser not implemented. Skipping: {file_path}")
#     return None

# def parse_cpp_ast(file_content, file_path):
#     """
#     Placeholder for a C++ AST parser.
#     You would need a library like 'clang'.
#     """
#     logging.warning(f"C++ AST parser not implemented. Skipping: {file_path}")
#     return None

# # This map links file extensions to their specific parser function
# PARSER_MAP = {
#     '.py': parse_python_ast,
#     '.java': parse_java_ast,
#     '.cpp': parse_cpp_ast,
#     '.c': parse_cpp_ast,
#     '.h': parse_cpp_ast,
# }


# def parse_file_to_ast(file_path):
#     """
#     Reads a file and routes it to the correct language-specific AST parser.

#     Input: A file path (str).
#     Output: An in-memory AST object, or None if parsing fails or is unsupported.
#     """
#     file_ext = os.path.splitext(file_path)[1].lower()
    
#     # Find the correct parser function from our map
#     parser_func = PARSER_MAP.get(file_ext)
    
#     if not parser_func:
#         logging.debug(f"No AST parser defined for extension {file_ext}. Skipping.")
#         return None

#     # Try to read the file content
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except (IOError, UnicodeDecodeError) as e:
#         logging.error(f"Could not read file {file_path}: {e}")
#         return None
        
#     # Call the specific parser with the file content
#     return parser_func(content, file_path)


# if __name__ == "__main__":
#     # --- Main Execution ---
    
#     # Set the target directory (same as in source_collector.py)
#     target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"
    
#     # --- Task 1.1: Run the collector ---
#     logging.info("--- [ Task 1.1: Collecting Source Files ] ---")
#     file_registry = scan_codebase(target_project_path)
    
#     if not file_registry:
#         logging.info("No files found. Exiting.")
#         exit()
        
#     logging.info(f"Found {len(file_registry)} files to parse.")

#     # --- Task 1.2: Run the parser ---
#     logging.info("\n--- [ Task 1.2: Parsing ASTs ] ---")
    
#     # This registry will hold our in-memory AST objects
#     # Key: file_path, Value: ast_object
#     ast_registry = {}

#     for file_record in file_registry:
#         file_path = file_record['path']
#         ast_obj = parse_file_to_ast(file_path)
        
#         if ast_obj:
#             ast_registry[file_path] = ast_obj

#     # --- Summary ---
#     logging.info("\n--- [ AST Parsing Summary ] ---")
#     logging.info(f"Total files scanned: {len(file_registry)}")
#     logging.info(f"Successfully parsed ASTs: {len(ast_registry)}")
    
#     unparsed_count = len(file_registry) - len(ast_registry)
#     if unparsed_count > 0:
#         logging.info(f"Skipped/Failed {unparsed_count} files (see warnings above).")


#     # --- [UPDATED] Example AST "Debug" ---
#     if ast_registry:
#         logging.info("\n--- [ Detailed AST 'Debug' for all Parsed Files ] ---")
#         try:
#             # Loop through ALL parsed files in the registry
#             for file_path, ast_obj in ast_registry.items():
                
#                 # We can only provide this debug for Python files
#                 if not file_path.endswith('.py'):
#                     logging.info(f"\n--- [ {file_path} ] ---")
#                     logging.info("  (Non-Python file, skipping detailed node dump)")
#                     continue

#                 logging.info(f"\n--- [ {file_path} ] ---")
                
#                 if not ast_obj.body:
#                     logging.info("  > This file appears to be empty.")
#                     continue

#                 # We can walk the top level of the AST 'body'
#                 found_node = False
#                 for top_level_node in ast_obj.body:
#                     found_node = True # Mark that we found something
                    
#                     if isinstance(top_level_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                         logging.info(f"    > Found Function: {top_level_node.name}")
                    
#                     elif isinstance(top_level_node, ast.ClassDef):
#                         logging.info(f"    > Found Class: {top_level_node.name}")
                    
#                     elif isinstance(top_level_node, ast.Import):
#                         modules = [alias.name for alias in top_level_node.names]
#                         logging.info(f"    > Found Import: {', '.join(modules)}")
                    
#                     elif isinstance(top_level_node, ast.ImportFrom):
#                         module_name = top_level_node.module or "..."
#                         logging.info(f"    > Found ImportFrom: {module_name}")
                    
#                     elif isinstance(top_level_node, ast.Assign):
#                         # This is for variables like API_KEY = "..."
#                         targets = [t.id for t in top_level_node.targets if isinstance(t, ast.Name)]
#                         if targets:
#                             logging.info(f"    > Found Global Variable: {', '.join(targets)}")
                
#                 if not found_node:
#                      logging.info("  > No top-level classes, functions, or imports found.")

#         except Exception as e:
#             logging.error(f"Error during AST example dump: {e}")




# import os
# import logging
# import ast  # Python's built-in Abstract Syntax Tree module

# # --- Task 1.1 Import ---
# # This script assumes 'source_collector.py' is in the same directory
# # We import the function and settings from your first file.
# try:
#     from source_collector import scan_codebase, SUPPORTED_EXTENSIONS
# except ImportError:
#     logging.error("FATAL: Could not import 'source_collector.py'.")
#     logging.error("Please make sure 'source_collector.py' is in the same folder.")
#     exit()
# # -----------------------


# # --- Configuration ---
# # Set up logging for this script.
# # NOTE: The logging config from the *imported* 'source_collector'
# # will likely take precedence. This is fine.
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [AST_PARSER] - %(message)s'
# )
# # --- End Configuration ---


# # --- Language-Specific Parsers ---

# def parse_python_ast(file_content, file_path):
#     """
#     Parses raw Python code into an AST object.
    
#     Input: Raw text content of a .py file.
#     Output: An in-memory AST object.
#     """
#     try:
#         ast_tree = ast.parse(file_content)
#         logging.info(f"Successfully parsed Python AST for: {file_path}")
#         return ast_tree
#     except SyntaxError as e:
#         logging.error(f"Syntax error in {file_path}: {e}")
#         return None
#     except Exception as e:
#         logging.error(f"Failed to parse {file_path}: {e}")
#         return None

# def parse_java_ast(file_content, file_path):
#     """
#     Placeholder for a Java AST parser.
#     You would need a library like 'plyj' or 'javalang'.
#     """
#     logging.warning(f"Java AST parser not implemented. Skipping: {file_path}")
#     return None

# def parse_cpp_ast(file_content, file_path):
#     """
#     Placeholder for a C++ AST parser.
#     You would need a library like 'clang'.
#     """
#     logging.warning(f"C++ AST parser not implemented. Skipping: {file_path}")
#     return None

# # This map links file extensions to their specific parser function
# PARSER_MAP = {
#     '.py': parse_python_ast,
#     '.java': parse_java_ast,
#     '.cpp': parse_cpp_ast,
#     '.c': parse_cpp_ast,
#     '.h': parse_cpp_ast,
# }


# def parse_file_to_ast(file_path):
#     """
#     Reads a file and routes it to the correct language-specific AST parser.

#     Input: A file path (str).
#     Output: An in-memory AST object, or None if parsing fails or is unsupported.
#     """
#     file_ext = os.path.splitext(file_path)[1].lower()
    
#     # Find the correct parser function from our map
#     parser_func = PARSER_MAP.get(file_ext)
    
#     if not parser_func:
#         logging.debug(f"No AST parser defined for extension {file_ext}. Skipping.")
#         return None

#     # Try to read the file content
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#     except (IOError, UnicodeDecodeError) as e:
#         logging.error(f"Could not read file {file_path}: {e}")
#         return None
        
#     # Call the specific parser with the file content
#     return parser_func(content, file_path)


# if __name__ == "__main__":
#     # --- Main Execution ---
    
#     # Set the target directory (same as in source_collector.py)
#     target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"
    
#     # --- Task 1.1: Run the collector ---
#     logging.info("--- [ Task 1.1: Collecting Source Files ] ---")
#     file_registry = scan_codebase(target_project_path)
    
#     if not file_registry:
#         logging.info("No files found. Exiting.")
#         exit()
        
#     logging.info(f"Found {len(file_registry)} files to parse.")

#     # --- Task 1.2: Run the parser ---
#     logging.info("\n--- [ Task 1.2: Parsing ASTs ] ---")
    
#     # This registry will hold our in-memory AST objects
#     # Key: file_path, Value: ast_object
#     ast_registry = {}

#     for file_record in file_registry:
#         file_path = file_record['path']
#         ast_obj = parse_file_to_ast(file_path)
        
#         if ast_obj:
#             ast_registry[file_path] = ast_obj

#     # --- Summary ---
#     logging.info("\n--- [ AST Parsing Summary ] ---")
#     logging.info(f"Total files scanned: {len(file_registry)}")
#     logging.info(f"Successfully parsed ASTs: {len(ast_registry)}")
    
#     unparsed_count = len(file_registry) - len(ast_registry)
#     if unparsed_count > 0:
#         logging.info(f"Skipped/Failed {unparsed_count} files (see warnings above).")


#     # --- [UPDATED] Example AST "Debug" ---
#     if ast_registry:
#         logging.info("\n--- [ Detailed AST 'Debug' for all Parsed Files ] ---")
#         try:
#             # Loop through ALL parsed files in the registry
#             for file_path, ast_obj in ast_registry.items():
                
#                 # We can only provide this debug for Python files
#                 if not file_path.endswith('.py'):
#                     logging.info(f"\n--- [ {file_path} ] ---")
#                     logging.info("  (Non-Python file, skipping detailed node dump)")
#                     continue

#                 logging.info(f"\n--- [ {file_path} ] ---")
                
#                 if not ast_obj.body:
#                     logging.info("  > This file appears to be empty.")
#                     continue

#                 # We can walk the top level of the AST 'body'
#                 found_node = False
#                 for top_level_node in ast_obj.body:
#                     found_node = True # Mark that we found something
                    
#                     if isinstance(top_level_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                         logging.info(f"    > Found Function: {top_level_node.name}")
                    
#                     elif isinstance(top_level_node, ast.ClassDef):
#                         logging.info(f"    > Found Class: {top_level_node.name}")
                        
#                         # --- [ NEW: Look inside the class ] ---
#                         class_body = top_level_node.body
#                         if not class_body:
#                             logging.info("        (Class appears to be empty)")
                            
#                         for class_node in class_body:
#                             if isinstance(class_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
#                                 # This is a "method"
#                                 logging.info(f"        -> Method: {class_node.name}")
#                             elif isinstance(class_node, ast.Assign):
#                                 # This is for class variables
#                                 class_vars = [t.id for t in class_node.targets if isinstance(t, ast.Name)]
#                                 if class_vars:
#                                     logging.info(f"        -> Class Variable: {', '.join(class_vars)}")
#                         # --- [ END NEW ] ---
                    
#                     elif isinstance(top_level_node, ast.Import):
#                         modules = [alias.name for alias in top_level_node.names]
#                         logging.info(f"    > Found Import: {', '.join(modules)}")
                    
#                     elif isinstance(top_level_node, ast.ImportFrom):
#                         module_name = top_level_node.module or "..."
#                         logging.info(f"    > Found ImportFrom: {module_name}")
                    
#                     elif isinstance(top_level_node, ast.Assign):
#                         # This is for variables like API_KEY = "..."
#                         targets = [t.id for t in top_level_node.targets if isinstance(t, ast.Name)]
#                         if targets:
#                             logging.info(f"    > Found Global Variable: {', '.join(targets)}")
                
#                 if not found_node:
#                      logging.info("  > No top-level classes, functions, or imports found.")

#         except Exception as e:
#             logging.error(f"Error during AST example dump: {e}")




import os
import logging
import ast # Python's built-in AST module

# Import the scanner from our first script
try:
    from source_collector import scan_codebase
except ImportError:
    logging.basicConfig() # Ensure logging is configured for the error
    logging.error("="*60)
    logging.error("FATAL: 'source_collector.py' not found.")
    logging.error("Please make sure all .py scripts are in the same directory.")
    logging.error("="*60)
    # We don't exit(1) here, so the importing script can handle the failure
    raise # Re-raise the ImportError

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [AST_PARSER] - %(message)s'
)

# The directory to scan, relative to this script's location
# You can change this to an absolute path if needed
TARGET_PROJECT_DIRECTORY = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"
# --- End Configuration ---


def parse_python_ast(file_path, file_content):
    """
    Parses raw Python code into an AST object.
    """
    try:
        return ast.parse(file_content, filename=file_path)
    except SyntaxError as e:
        logging.warning(f"SyntaxError in {file_path} at line {e.lineno}: {e.msg}")
        return None
    except Exception as e:
        logging.error(f"Failed to parse Python AST for {file_path}: {e}")
        return None

def parse_java_ast(file_path, file_content):
    """
    (Placeholder) Parses raw Java code into an AST object.
    """
    logging.warning(f"Java parser not implemented. Skipping: {file_path}")
    # Example (if you install 'javalang'):
    # try:
    #     import javalang
    #     return javalang.parse.parse(file_content)
    # except Exception as e:
    #     logging.error(f"Failed to parse Java AST for {file_path}: {e}")
    #     return None
    return None

def parse_file_to_ast(file_path):
    """
    Reads a file and routes it to the correct language-specific parser.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None

    # Route to the correct parser based on extension
    if extension == '.py':
        ast_obj = parse_python_ast(file_path, content)
        if ast_obj:
            logging.info(f"Successfully parsed Python AST for: {file_path}")
        return ast_obj
        
    elif extension == '.java':
        return parse_java_ast(file_path, content)
        
    # ... add more elif blocks for other languages ...
    
    else:
        logging.debug(f"No parser available for extension {extension}. Skipping: {file_path}")
        return None

def parse_code_to_asts(start_directory=TARGET_PROJECT_DIRECTORY):
    """
    Runs the full Task 1.1 and 1.2 pipeline.
    This is the main importable function for node_extractor.py
    
    Returns:
        A dictionary mapping {file_path: ast_object}
    """
    
    # --- Task 1.1 ---
    logging.info("--- [ Running Task 1.1: Collecting Source Files ] ---")
    file_registry = scan_codebase(start_directory)
    if not file_registry:
        logging.warning("No source files found by scanner.")
        return {}
    
    logging.info(f"Found {len(file_registry)} files to parse.")
    
    # --- Task 1.2 ---
    logging.info("--- [ Running Task 1.2: Parsing ASTs ] ---")
    ast_registry = {} # {file_path: ast_object}
    
    for file_record in file_registry:
        file_path = file_record['path']
        ast_obj = parse_file_to_ast(file_path)
        if ast_obj:
            ast_registry[file_path] = ast_obj
            
    logging.info(f"Successfully parsed {len(ast_registry)} / {len(file_registry)} files.")
    return ast_registry

# This script is now a module.
# The 'node_extractor.py' script will call 'parse_code_to_asts()'

