# import os
# import logging
# import ast

# # --- Task 1.2 Import ---
# # This script assumes 'ast_parser.py' is in the same directory
# try:
#     # We will "run" the ast_parser's main block as a function
#     # To do this, we need to refactor ast_parser.py slightly
#     # (See instructions after this code block)
    
#     # For now, let's assume we can import a "main" function
#     # that returns the registry
#     from ast_parser import parse_file_to_ast
#     from source_collector import scan_codebase

# except ImportError:
#     logging.error("FATAL: Could not import from 'ast_parser.py' or 'source_collector.py'.")
#     logging.error("Please make sure all scripts are in the same folder.")
#     exit()
# # -----------------------

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
# )
# # --- End Configuration ---


# class HierarchicalNodeVisitor(ast.NodeVisitor):
#     """
#     An AST visitor that extracts key architectural nodes.
    
#     This visitor traverses the *full* AST and builds a *simplified*
#     list containing only the "abstraction" nodes we care about
#     (classes, functions, and methods).
#     """
#     def __init__(self):
#         self.nodes = []
#         self.current_class_name = None

#     def visit_ClassDef(self, node):
#         """
#         Called when the visitor finds a class.
#         """
#         logging.debug(f"Found Class: {node.name}")
        
#         # We store the node itself, along with its type and file-level context
#         self.nodes.append({
#             "type": "ClassDef",
#             "name": node.name,
#             "node_object": node
#         })
        
#         # Store the class name so methods know their parent
#         self.current_class_name = node.name
#         # Continue traversing *inside* the class to find methods
#         self.generic_visit(node)
#         # We're leaving the class, so unset the name
#         self.current_class_name = None

#     def visit_FunctionDef(self, node):
#         """
#         Called when the visitor finds a function (or method).
#         """
#         if self.current_class_name:
#             # This is a method
#             node_type = "MethodDef"
#             full_name = f"{self.current_class_name}.{node.name}"
#             logging.debug(f"Found Method: {full_name}")
#         else:
#             # This is a top-level function
#             node_type = "FunctionDef"
#             full_name = node.name
#             logging.debug(f"Found Function: {full_name}")

#         self.nodes.append({
#             "type": node_type,
#             "name": full_name,
#             "node_object": node
#         })
#         # **Crucial**: We stop traversing here.
#         # We don't care about 'if' statements or 'for' loops *inside*
#         # the function. The function itself is the "chunk".
#         # Do not call self.generic_visit(node)

#     def visit_AsyncFunctionDef(self, node):
#         """
#         Called when the visitor finds an async function (or method).
#         """
#         # Treat async functions just like regular functions
#         self.visit_FunctionDef(node)


# def extract_hierarchical_nodes(ast_registry):
#     """
#     Takes the full AST registry and filters it down to a simplified
#     list of key nodes for each file.

#     Input: { file_path: full_ast_object }
#     Output: { file_path: [ list_of_key_node_dicts ] }
#     """
    
#     hierarchical_node_registry = {}
    
#     logging.info(f"Extracting key nodes from {len(ast_registry)} ASTs...")
    
#     for file_path, ast_obj in ast_registry.items():
#         if not ast_obj:
#             continue
            
#         logging.debug(f"Visiting AST for: {file_path}")
        
#         # 1. Create a new visitor instance for each file
#         visitor = HierarchicalNodeVisitor()
        
#         # 2. Traverse the full AST. The visitor will
#         # automatically collect the nodes it cares about.
#         visitor.visit(ast_obj)
        
#         # 3. Store the simplified list of nodes
#         hierarchical_node_registry[file_path] = visitor.nodes
        
#         logging.debug(f"Extracted {len(visitor.nodes)} key nodes from: {file_path}")

#     return hierarchical_node_registry


# def get_ast_registry(start_directory):
#     """
#     Runs the full Task 1.1 and 1.2 pipeline.
#     """
#     logging.info("--- [ Running Task 1.1: Collecting Source Files ] ---")
#     file_registry = scan_codebase(start_directory)
#     if not file_registry:
#         logging.info("No files found.")
#         return {}
#     logging.info(f"Found {len(file_registry)} files.")
    
#     logging.info("\n--- [ Running Task 1.2: Parsing ASTs ] ---")
#     ast_registry = {}
#     for file_record in file_registry:
#         file_path = file_record['path']
#         ast_obj = parse_file_to_ast(file_path)
#         if ast_obj:
#             ast_registry[file_path] = ast_obj
#     logging.info(f"Successfully parsed {len(ast_registry)} ASTs.")
#     return ast_registry


# if __name__ == "__main__":
#     # --- Main Execution ---
    
#     target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"

#     # --- Run Tasks 1.1 and 1.2 ---
#     full_ast_registry = get_ast_registry(target_project_path)
    
#     if not full_ast_registry:
#         logging.info("Exiting.")
#         exit()

#     # --- Task 1.3: Run the Extractor ---
#     logging.info("\n--- [ Task 1.3: Extracting Hierarchical Nodes ] ---")
    
#     # This is the "simplified list of code chunks"
#     hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)

#     # --- Summary & Debugging ---
#     logging.info("\n--- [ Node Extraction Summary ] ---")
#     logging.info(f"Processed {len(hierarchical_node_registry)} files.")
    
#     total_nodes = 0
#     for file_path, nodes in hierarchical_node_registry.items():
#         total_nodes += len(nodes)
        
#     logging.info(f"Total key nodes (chunks) extracted: {total_nodes}")

#     # --- Detailed "Debug" Print ---
#     if hierarchical_node_registry:
#         logging.info("\n--- [ Detailed Hierarchical Node Registry ] ---")
#         for file_path, nodes in hierarchical_node_registry.items():
#             short_path = os.path.basename(file_path)
#             logging.info(f"\n--- [ {short_path} ] ---")
#             if not nodes:
#                 logging.info("  (No classes or functions found)")
#                 continue
                
#             for node_info in nodes:
#                 logging.info(f"  > Found {node_info['type']}: {node_info['name']}")






# import os
# import logging
# import ast
# import json  # Added for JSON export

# # --- Task 1.2 Import ---
# # ... (rest of the import block is unchanged) ...
# try:
#     # We will "run" the ast_parser's main block as a function
#     # To do this, we need to refactor ast_parser.py slightly
#     # (See instructions after this code block)
    
#     # For now, let's assume we can import a "main" function
#     # that returns the registry
#     from ast_parser import parse_file_to_ast
#     from source_collector import scan_codebase

# except ImportError:
#     logging.error("FATAL: Could not import from 'ast_parser.py' or 'source_collector.py'.")
#     logging.error("Please make sure all scripts are in the same folder.")
#     exit()
# # -----------------------

# # --- Configuration ---
# # ... (logging config is unchanged) ...
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
# )
# # --- End Configuration ---


# class HierarchicalNodeVisitor(ast.NodeVisitor):
#     """
#     An AST visitor that extracts key architectural nodes.
    
#     This visitor traverses the *full* AST and builds a *simplified*
#     list containing only the "abstraction" nodes we care about
#     (classes, functions, and methods).
#     """
#     def __init__(self):
#         self.nodes = []
#         self.current_class_name = None

#     def visit_ClassDef(self, node):
#         """
#         Called when the visitor finds a class.
#         """
#         logging.debug(f"Found Class: {node.name}")
        
#         # We store the node itself, along with its type and file-level context
#         self.nodes.append({
#             "type": "ClassDef",
#             "name": node.name,
#             "node_object": node  # We keep the object for potential future steps
#         })
        
#         # Store the class name so methods know their parent
#         self.current_class_name = node.name
#         # Continue traversing *inside* the class to find methods
#         self.generic_visit(node)
#         # We're leaving the class, so unset the name
#         self.current_class_name = None

#     def visit_FunctionDef(self, node):
#         """
#         Called when the visitor finds a function (or method).
#         """
#         if self.current_class_name:
#             # This is a method
#             node_type = "MethodDef"
#             full_name = f"{self.current_class_name}.{node.name}"
#             logging.debug(f"Found Method: {full_name}")
#         else:
#             # This is a top-level function
#             node_type = "FunctionDef"
#             full_name = node.name
#             logging.debug(f"Found Function: {full_name}")

#         self.nodes.append({
#             "type": node_type,
#             "name": full_name,
#             "node_object": node # We keep the object for potential future steps
#         })
#         # **Crucial**: We stop traversing here.
#         # We don't care about 'if' statements or 'for' loops *inside*
#         # the function. The function itself is the "chunk".
#         # Do not call self.generic_visit(node)

#     def visit_AsyncFunctionDef(self, node):
#         """
#         Called when the visitor finds an async function (or method).
#         """
#         # Treat async functions just like regular functions
#         self.visit_FunctionDef(node)


# def extract_hierarchical_nodes(ast_registry):
#     """
#     Takes the full AST registry and filters it down to a simplified
#     list of key nodes for each file.

#     Input: { file_path: full_ast_object }
#     Output: { file_path: [ list_of_key_node_dicts ] }
#     """
    
#     hierarchical_node_registry = {}
    
#     logging.info(f"Extracting key nodes from {len(ast_registry)} ASTs...")
    
#     for file_path, ast_obj in ast_registry.items():
#         if not ast_obj:
#             continue
            
#         logging.debug(f"Visiting AST for: {file_path}")
        
#         # 1. Create a new visitor instance for each file
#         visitor = HierarchicalNodeVisitor()
        
#         # 2. Traverse the full AST. The visitor will
#         # automatically collect the nodes it cares about.
#         visitor.visit(ast_obj)
        
#         # 3. Store the simplified list of nodes
#         hierarchical_node_registry[file_path] = visitor.nodes
        
#         logging.debug(f"Extracted {len(visitor.nodes)} key nodes from: {file_path}")

#     return hierarchical_node_registry


# def get_ast_registry(start_directory):
#     """
#     Runs the full Task 1.1 and 1.2 pipeline.
#     """
#     # ... (this function is unchanged) ...
#     logging.info("--- [ Running Task 1.1: Collecting Source Files ] ---")
#     file_registry = scan_codebase(start_directory)
#     if not file_registry:
#         logging.info("No files found.")
#         return {}
#     logging.info(f"Found {len(file_registry)} files.")
    
#     logging.info("\n--- [ Running Task 1.2: Parsing ASTs ] ---")
#     ast_registry = {}
#     for file_record in file_registry:
#         file_path = file_record['path']
#         ast_obj = parse_file_to_ast(file_path)
#         if ast_obj:
#             ast_registry[file_path] = ast_obj
#     logging.info(f"Successfully parsed {len(ast_registry)} ASTs.")
#     return ast_registry


# # --- [ NEW FUNCTION FOR JSON EXPORT ] ---
# def export_registry_to_json(registry, output_filename):
#     """
#     Exports the hierarchical registry to a JSON file.
#     It creates a copy of the registry without the non-serializable
#     'node_object' items.
#     """
#     logging.info(f"\n--- [ Exporting Registry to {output_filename} ] ---")
    
#     serializable_registry = {}
    
#     for file_path, nodes in registry.items():
#         serializable_nodes = []
#         for node_info in nodes:
#             # Create a copy and remove the AST object
#             # which is not JSON-serializable
#             serializable_info = node_info.copy()
#             del serializable_info['node_object']
#             serializable_nodes.append(serializable_info)
        
#         # Use relative path for cleaner JSON
#         relative_path = os.path.relpath(file_path)
#         serializable_registry[relative_path] = serializable_nodes

#     try:
#         with open(output_filename, 'w', encoding='utf-8') as f:
#             json.dump(serializable_registry, f, indent=4)
#         logging.info(f"Successfully exported node registry to {output_filename}")
#     except (IOError, TypeError) as e:
#         logging.error(f"Failed to export JSON file: {e}")
# # --- [ END NEW FUNCTION ] ---


# if __name__ == "__main__":
#     # --- Main Execution ---
    
#     target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"

#     # --- Run Tasks 1.1 and 1.2 ---
#     full_ast_registry = get_ast_registry(target_project_path)
    
#     if not full_ast_registry:
#         logging.info("Exiting.")
#         exit()

#     # --- Task 1.3: Run the Extractor ---
#     logging.info("\n--- [ Task 1.3: Extracting Hierarchical Nodes ] ---")
    
#     # This is the "simplified list of code chunks"
#     hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)

#     # --- Summary & Debugging ---
#     logging.info("\n--- [ Node Extraction Summary ] ---")
#     # ... (summary logging is unchanged) ...
#     logging.info(f"Processed {len(hierarchical_node_registry)} files.")
    
#     total_nodes = 0
#     for file_path, nodes in hierarchical_node_registry.items():
#         total_nodes += len(nodes)
        
#     logging.info(f"Total key nodes (chunks) extracted: {total_nodes}")

#     # --- Detailed "Debug" Print ---
#     if hierarchical_node_registry:
#         logging.info("\n--- [ Detailed Hierarchical Node Registry ] ---")
#         # ... (detailed logging is unchanged) ...
#         for file_path, nodes in hierarchical_node_registry.items():
#             short_path = os.path.basename(file_path)
#             logging.info(f"\n--- [ {short_path} ] ---")
#             if not nodes:
#                 logging.info("  (No classes or functions found)")
#                 continue
                
#             for node_info in nodes:
#                 logging.info(f"  > Found {node_info['type']}: {node_info['name']}")

#     # --- [ NEW: Export the result to JSON ] ---
#     if hierarchical_node_registry:
#         export_registry_to_json(hierarchical_node_registry, 'hierarchical_nodes.json')




# import logging
# import sys
# import json
# import ast # <-- NEW: To convert AST nodes back to text

# # Import the functions from our previous scripts
# try:
#     from ast_parser import parse_code_to_asts
#     from anonymizer import anonymize_code_chunk # <-- NEW: Import anonymizer
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'ast_parser.py' or 'anonymizer.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
# )
# # --- End Configuration ---

# class HierarchicalNodeVisitor(ast.NodeVisitor):
#     """
#     A custom AST visitor that extracts only high-level abstraction
#     nodes (Classes, Functions) and their source code.
#     """
#     def __init__(self):
#         # This list will hold the extracted "chunks"
#         self.nodes = []

#     def visit_ClassDef(self, node):
#         """Called when a 'class ...' definition is found."""
#         logging.debug(f"Found ClassDef node: {node.name}")
        
#         # --- [ NEW LOGIC ] ---
#         # 1. Convert the AST node back into its original source code
#         source_code = ast.unparse(node)
#         # 2. Anonymize this specific code chunk
#         anonymized_code = anonymize_code_chunk(source_code)
#         # 3. Save the important info
#         self.nodes.append({
#             'type': 'ClassDef',
#             'name': node.name,
#             'anonymized_source_code': anonymized_code
#         })
#         # 4. Stop traversing deeper into this class's methods
#         #    (The main loop will visit them separately)
#         #    We do this by *not* calling generic_visit(node)

#     def visit_FunctionDef(self, node):
#         """Called when a 'def ...' function is found."""
#         logging.debug(f"Found FunctionDef node: {node.name}")
        
#         # --- [ NEW LOGIC ] ---
#         source_code = ast.unparse(node)
#         anonymized_code = anonymize_code_chunk(source_code)
#         self.nodes.append({
#             'type': 'FunctionDef',
#             'name': node.name,
#             'anonymized_source_code': anonymized_code
#         })
#         # Stop traversing into the function's body
        
#     def visit_AsyncFunctionDef(self, node):
#         """Called when an 'async def ...' function is found."""
#         logging.debug(f"Found AsyncFunctionDef node: {node.name}")
        
#         # --- [ NEW LOGIC ] ---
#         source_code = ast.unparse(node)
#         anonymized_code = anonymize_code_chunk(source_code)
#         self.nodes.append({
#             'type': 'AsyncFunctionDef',
#             'name': node.name,
#             'anonymized_source_code': anonymized_code
#         })
#         # Stop traversing into the function's body

#     def generic_visit(self, node):
#         """
#         The fallback visitor. We only call this *if* the node
#         is not one we're stopping at (like a Module or a simple
#         import), allowing it to continue traversing down the tree.
#         """
#         if node is not None:
#             super().generic_visit(node)


# def extract_hierarchical_nodes(full_ast_registry: dict) -> dict:
#     """
#     Traverses the full AST registry and extracts only the
#     hierarchical nodes (classes, functions).
    
#     Args:
#         full_ast_registry: The dict from ast_parser.py {filepath: ast_object}

#     Returns:
#         A new registry: {filepath: [list_of_node_dicts]}
#     """
#     logging.info("--- [ Task 1.3: Hierarchical Node Identification ] ---")
#     hierarchical_registry = {}
    
#     for file_path, ast_object in full_ast_registry.items():
#         if ast_object:
#             visitor = HierarchicalNodeVisitor()
#             # Start the traversal
#             visitor.visit(ast_object)
            
#             if visitor.nodes:
#                 logging.info(f"Extracted {len(visitor.nodes)} abstraction nodes from: {file_path}")
#                 hierarchical_registry[file_path] = visitor.nodes
#             else:
#                 logging.info(f"No high-level nodes found in: {file_path}")
#         else:
#             logging.warning(f"Skipping {file_path}, no AST object found.")
            
#     return hierarchical_registry


# def export_registry_to_json(registry: dict, output_filename: str):
#     """
#     Exports the hierarchical node registry to a JSON file.
#     The registry is now serializable by default.
#     """
#     logging.info(f"\nExporting hierarchical node registry to: {output_filename}")
#     try:
#         with open(output_filename, 'w', encoding='utf-8') as f:
#             # We can now just dump the registry directly
#             json.dump(registry, f, indent=2)
#         logging.info(f"Successfully created '{output_filename}'")
#     except TypeError as e:
#         logging.error(f"Failed to serialize registry to JSON: {e}")
#     except IOError as e:
#         logging.error(f"Failed to write JSON file: {e}")
        
# # --- [ Main execution ] ---
# def main():
#     """
#     Runs the full pipeline from parsing to node extraction
#     and exports the final "code chunk" registry.
#     """
#     # Run Task 1.2
#     full_ast_registry = parse_code_to_asts()
    
#     if not full_ast_registry:
#         logging.warning("AST parsing returned no results. Exiting.")
#         return

#     # Run Task 1.3
#     hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)
    
#     if not hierarchical_node_registry:
#         logging.warning("Node extraction returned no results. Exiting.")
#         return

#     # Export the final "chunks" to JSON for the next phase
#     export_registry_to_json(hierarchical_node_registry, 'hierarchical_nodes.json')
    
#     logging.info("\n--- [ Node Extraction Summary ] ---")
#     total_nodes = 0
#     for file_path, nodes in hierarchical_node_registry.items():
#         logging.info(f"\nFile: {file_path}")
#         for node in nodes:
#             logging.info(f"  - {node['type']}: {node['name']}")
#             total_nodes += 1
#     logging.info(f"\nTotal abstraction nodes (chunks) identified: {total_nodes}")


# if __name__ == "__main__":
#     # Note: Requires Python 3.9+ for ast.unparse()
#     if sys.version_info < (3, 9):
#         logging.error("="*60)
#         logging.error("FATAL: This script requires Python 3.9+ for 'ast.unparse()'")
#         logging.error("Please upgrade your Python version.")
#         logging.error("="*60)
#         sys.exit(1)
        
#     main()





# import logging
# import sys
# import json
# import ast

# # Import the functions from our previous scripts
# try:
#     from ast_parser import parse_code_to_asts
#     from anonymizer import anonymize_code_chunk
# except ImportError:
#     logging.basicConfig() # Ensure logging is configured for the error
#     logging.error("="*60)
#     logging.error("FATAL: 'ast_parser.py' or 'anonymizer.py' not found.")
#     logging.error("Please make sure all .py scripts are in the same directory.")
#     logging.error("="*60)
#     sys.exit(1) # Exit if the dependency is missing

# # --- Configuration ---
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
# )
# # --- End Configuration ---

# class HierarchicalNodeVisitor(ast.NodeVisitor):
#     """
#     A custom AST visitor that extracts high-level abstraction
#     nodes (Classes, Functions, and Methods) as individual "chunks".
    
#     This is the corrected logic.
#     """
#     def __init__(self):
#         # This list will hold the extracted "chunks"
#         self.nodes = []
#         self.current_class_name = None # Used to track methods

#     def visit_ClassDef(self, node):
#         """Called when a 'class ...' definition is found."""
#         logging.debug(f"Found ClassDef node: {node.name}")

#         # --- [ CORRECTED LOGIC ] ---
#         # 1. We create a "shell" node for the class definition itself.
#         #    We use ast.unparse() on a *copy* of the node with an empty body
#         #    to just get the 'class MyClass(Base):' part.
#         class_shell = ast.ClassDef(
#             name=node.name,
#             bases=node.bases,
#             keywords=node.keywords,
#             decorator_list=node.decorator_list,
#             body=[] # Empty body
#         )
#         source_code = ast.unparse(class_shell)
#         anonymized_code = anonymize_code_chunk(source_code)
        
#         # 2. Save the Class "chunk"
#         self.nodes.append({
#             'type': 'ClassDef',
#             'name': node.name,
#             'anonymized_source_code': anonymized_code
#         })
        
#         # 3. Now, traverse *into* the class to find its methods
#         self.current_class_name = node.name
#         self.generic_visit(node)
#         self.current_class_name = None # Leaving the class

#     def visit_FunctionDef(self, node):
#         """Called when a 'def ...' function or method is found."""
        
#         # Determine if it's a Method or a top-level Function
#         if self.current_class_name:
#             node_type = "MethodDef"
#             full_name = f"{self.current_class_name}.{node.name}"
#             logging.debug(f"Found MethodDef node: {full_name}")
#         else:
#             node_type = "FunctionDef"
#             full_name = node.name
#             logging.debug(f"Found FunctionDef node: {full_name}")

#         # --- [ CORRECTED LOGIC ] ---
#         # 1. Unparse the *entire function* as its own chunk
#         source_code = ast.unparse(node)
#         # 2. Anonymize this specific code chunk
#         anonymized_code = anonymize_code_chunk(source_code)
        
#         # 3. Save the Function/Method "chunk"
#         self.nodes.append({
#             'type': node_type,
#             'name': full_name,
#             'anonymized_source_code': anonymized_code
#         })
#         # 4. **Crucial**: We *stop* traversing here.
#         #    We don't care about 'if's or 'for's *inside* the function.
#         #    The function itself is the chunk.
#         #    (i.e., we do NOT call self.generic_visit(node))

#     def visit_AsyncFunctionDef(self, node):
#         """Called when an 'async def ...' function or method is found."""
#         # Treat it just like a regular function
#         self.visit_FunctionDef(node)
        
#     # We no longer need a 'generic_visit' override,
#     # as the default behavior is now correct.


# def extract_hierarchical_nodes(full_ast_registry: dict) -> dict:
#     """
#     Traverses the full AST registry and extracts the
#     hierarchical nodes (classes, functions, methods).
#     """
#     logging.info("--- [ Task 1.3: Hierarchical Node Identification ] ---")
#     hierarchical_registry = {}
    
#     for file_path, ast_object in full_ast_registry.items():
#         if ast_object:
#             visitor = HierarchicalNodeVisitor()
#             visitor.visit(ast_object)
            
#             if visitor.nodes:
#                 logging.info(f"Extracted {len(visitor.nodes)} abstraction nodes from: {file_path}")
#                 hierarchical_registry[file_path] = visitor.nodes
#             else:
#                 logging.info(f"No high-level nodes found in: {file_path}")
#         else:
#             logging.warning(f"Skipping {file_path}, no AST object found.")
            
#     return hierarchical_registry


# def export_registry_to_json(registry: dict, output_filename: str):
#     """
#     Exports the hierarchical node registry to a JSON file.
#     """
#     logging.info(f"\nExporting hierarchical node registry to: {output_filename}")
#     try:
#         with open(output_filename, 'w', encoding='utf-8') as f:
#             json.dump(registry, f, indent=2)
#         logging.info(f"Successfully created '{output_filename}'")
#     except TypeError as e:
#         logging.error(f"Failed to serialize registry to JSON: {e}")
#     except IOError as e:
#         logging.error(f"Failed to write JSON file: {e}")
        
# # --- [ Main execution ] ---
# def main():
#     """
#     Runs the full pipeline from parsing to node extraction
#     and exports the final "code chunk" registry.
#     """
#     # Run Task 1.2
#     full_ast_registry = parse_code_to_asts()
    
#     if not full_ast_registry:
#         logging.warning("AST parsing returned no results. Exiting.")
#         return

#     # Run Task 1.3
#     hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)
    
#     if not hierarchical_node_registry:
#         logging.warning("Node extraction returned no results. Exiting.")
#         return

#     # Export the final "chunks" to JSON for the next phase
#     export_registry_to_json(hierarchical_node_registry, 'hierarchical_nodes.json')
    
#     logging.info("\n--- [ Node Extraction Summary ] ---")
#     total_nodes = 0
#     for file_path, nodes in hierarchical_node_registry.items():
#         logging.info(f"\nFile: {file_path}")
#         for node in nodes:
#             logging.info(f"  - {node['type']}: {node['name']}")
#             total_nodes += 1
#     logging.info(f"\nTotal abstraction nodes (chunks) identified: {total_nodes}")


# if __name__ == "__main__":
#     if sys.version_info < (3, 9):
#         logging.error("="*60)
#         logging.error("FATAL: This script requires Python 3.9+ for 'ast.unparse()'")
#         logging.error("Please upgrade your Python version.")
#         logging.error("="*60)
#         sys.exit(1)
        
#     main()








import os
import logging
import ast
import json
import sys

# --- Task 1.2 Import ---
try:
    from ast_parser import parse_file_to_ast
    from source_collector import scan_codebase
except ImportError:
    logging.basicConfig() # Ensure logging is configured for the error
    logging.error("="*60)
    logging.error("FATAL: 'ast_parser.py' or 'source_collector.py' not found.")
    logging.error("Please make sure all .py scripts are in the same directory.")
    logging.error("="*60)
    sys.exit(1)

# --- [CRITICAL FIX] Task 2.2 Import ---
# We MUST import the anonymizer here to clean the code chunks.
try:
    from anonymizer import anonymize_code_chunk
except ImportError:
    logging.basicConfig()
    logging.error("="*60)
    logging.error("FATAL: 'anonymizer.py' not found.")
    logging.error("Please make sure 'anonymizer.py' is in the same directory.")
    logging.error("="*60)
    sys.exit(1)
# ------------------------------------

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
)
OUTPUT_FILE = "hierarchical_nodes.json"
# --- End Configuration ---


class HierarchicalNodeVisitor(ast.NodeVisitor):
    """
    An AST visitor that extracts key architectural nodes (Classes and Methods)
    and their *anonymized* source code.
    """
    def __init__(self):
        self.nodes = []
        self.current_class_name = None

    def visit_ClassDef(self, node):
        """
        Called when the visitor finds a class.
        We extract the class definition as a "chunk".
        """
        logging.debug(f"Found Class: {node.name}")
        
        # --- [CRITICAL FIX] ---
        # 1. Unparse the node to get its raw source code
        # We create a "dummy" node with just the class signature
        # as we don't want the full class body, just its definition.
        class_signature_node = ast.ClassDef(
            name=node.name,
            bases=node.bases,
            keywords=node.keywords,
            decorator_list=node.decorator_list,
            body=[] # Empty body
        )
        try:
            # unparse() converts the AST node back into text code
            source_code = ast.unparse(class_signature_node)
        except Exception as e:
            logging.warning(f"Could not unparse ClassDef '{node.name}': {e}")
            source_code = f"class {node.name}: ..."

        # 2. Anonymize the source code (e.g., in decorators)
        anonymized_code = anonymize_code_chunk(source_code)
        # --- [END FIX] ---

        self.nodes.append({
            "type": "ClassDef",
            "name": node.name,
            "anonymized_source_code": anonymized_code # Store the clean code
        })
        
        # Store the class name so methods know their parent
        self.current_class_name = node.name
        # Continue traversing *inside* the class to find methods
        self.generic_visit(node)
        # We're leaving the class, so unset the name
        self.current_class_name = None

    def visit_FunctionDef(self, node):
        """
        Called when the visitor finds a function (or method).
        This is treated as one "chunk".
        """
        if self.current_class_name:
            node_type = "MethodDef"
            full_name = f"{self.current_class_name}.{node.name}"
            logging.debug(f"Found Method: {full_name}")
        else:
            node_type = "FunctionDef"
            full_name = node.name
            logging.debug(f"Found Function: {full_name}")

        # --- [CRITICAL FIX] ---
        # 1. Unparse the node to get its raw source code
        try:
            # unparse() converts the AST node back into text code
            source_code = ast.unparse(node)
        except Exception as e:
            logging.warning(f"Could not unparse {node_type} '{full_name}': {e}")
            source_code = f"def {node.name}(...): ..."
            
        # 2. Anonymize the source code (this is the key step)
        anonymized_code = anonymize_code_chunk(source_code)
        # --- [END FIX] ---

        self.nodes.append({
            "type": node_type,
            "name": full_name,
            "anonymized_source_code": anonymized_code # Store the clean code
        })
        # **Crucial**: We stop traversing here.
        # The function itself is the "chunk".
        # Do not call self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """
        Called when the visitor finds an async function (or method).
        """
        # Treat async functions just like regular functions
        self.visit_FunctionDef(node)


def extract_hierarchical_nodes(ast_registry):
    """
    Takes the full AST registry and filters it down to a simplified
    list of key nodes for each file.
    """
    hierarchical_node_registry = {}
    logging.info(f"Extracting key nodes from {len(ast_registry)} ASTs...")
    
    for file_path, ast_obj in ast_registry.items():
        if not ast_obj:
            continue
            
        logging.debug(f"Visiting AST for: {file_path}")
        visitor = HierarchicalNodeVisitor()
        visitor.visit(ast_obj)
        
        hierarchical_node_registry[file_path] = visitor.nodes
        logging.debug(f"Extracted {len(visitor.nodes)} key nodes from: {file_path}")

    return hierarchical_node_registry


def get_ast_registry(start_directory):
    """
    Runs the full Task 1.1 and 1.2 pipeline.
    """
    logging.info("--- [ Running Task 1.1: Collecting Source Files ] ---")
    file_registry = scan_codebase(start_directory)
    if not file_registry:
        logging.info("No files found.")
        return {}
    logging.info(f"Found {len(file_registry)} files.")
    
    logging.info("\n--- [ Running Task 1.2: Parsing ASTs ] ---")
    ast_registry = {}
    for file_record in file_registry:
        file_path = file_record['path']
        ast_obj = parse_file_to_ast(file_path)
        if ast_obj:
            ast_registry[file_path] = ast_obj
    logging.info(f"Successfully parsed {len(ast_registry)} ASTs.")
    return ast_registry


def export_registry_to_json(registry, output_filename):
    """
    Exports the hierarchical registry to a JSON file.
    """
    logging.info(f"\n--- [ Exporting Registry to {output_filename} ] ---")
    
    serializable_registry = {}
    for file_path, nodes in registry.items():
        # Use relative path for cleaner JSON
        relative_path = os.path.relpath(file_path)
        serializable_registry[relative_path] = nodes

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_registry, f, indent=4)
        logging.info(f"Successfully exported node registry to {output_filename}")
    except (IOError, TypeError) as e:
        logging.error(f"Failed to export JSON file: {e}")


def main():
    """
    Main execution function.
    """
    # Define the target project directory
    # --- [ IMPORTANT ] ---
    # You MUST update this path to point to your 'Shopping' folder
    # We use an 'r' string (raw string) to handle backslashes correctly
    target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"
    # --- [ END IMPORTANT ] ---

    if not os.path.isdir(target_project_path):
        logging.error(f"FATAL: Target project path not found:")
        logging.error(f"{target_project_path}")
        logging.error("Please update the 'target_project_path' variable in this script.")
        sys.exit(1)

    # --- Run Tasks 1.1 and 1.2 ---
    full_ast_registry = get_ast_registry(target_project_path)
    
    if not full_ast_registry:
        logging.info("No ASTs were parsed. Exiting.")
        sys.exit()

    # --- Task 1.3: Run the Extractor ---
    logging.info("\n--- [ Task 1.3: Extracting Hierarchical Nodes ] ---")
    hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)

    # --- Summary ---
    logging.info("\n--- [ Node Extraction Summary ] ---")
    logging.info(f"Processed {len(hierarchical_node_registry)} files.")
    total_nodes = sum(len(nodes) for nodes in hierarchical_node_registry.values())
    logging.info(f"Total key nodes (chunks) extracted: {total_nodes}")

    # --- [ NEW: Export the result to JSON ] ---
    if hierarchical_node_registry:
        export_registry_to_json(hierarchical_node_registry, OUTPUT_FILE)
    
    logging.info("\n--- [ node_extractor.py finished ] ---")
    logging.info(f"Output file '{OUTPUT_FILE}' is ready for 'smart_summarizer.py'.")

if __name__ == "__main__":
    main()



