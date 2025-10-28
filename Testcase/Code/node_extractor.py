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






import os
import logging
import ast
import json  # Added for JSON export

# --- Task 1.2 Import ---
# ... (rest of the import block is unchanged) ...
try:
    # We will "run" the ast_parser's main block as a function
    # To do this, we need to refactor ast_parser.py slightly
    # (See instructions after this code block)
    
    # For now, let's assume we can import a "main" function
    # that returns the registry
    from ast_parser import parse_file_to_ast
    from source_collector import scan_codebase

except ImportError:
    logging.error("FATAL: Could not import from 'ast_parser.py' or 'source_collector.py'.")
    logging.error("Please make sure all scripts are in the same folder.")
    exit()
# -----------------------

# --- Configuration ---
# ... (logging config is unchanged) ...
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [NODE_EXTRACTOR] - %(message)s'
)
# --- End Configuration ---


class HierarchicalNodeVisitor(ast.NodeVisitor):
    """
    An AST visitor that extracts key architectural nodes.
    
    This visitor traverses the *full* AST and builds a *simplified*
    list containing only the "abstraction" nodes we care about
    (classes, functions, and methods).
    """
    def __init__(self):
        self.nodes = []
        self.current_class_name = None

    def visit_ClassDef(self, node):
        """
        Called when the visitor finds a class.
        """
        logging.debug(f"Found Class: {node.name}")
        
        # We store the node itself, along with its type and file-level context
        self.nodes.append({
            "type": "ClassDef",
            "name": node.name,
            "node_object": node  # We keep the object for potential future steps
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
        """
        if self.current_class_name:
            # This is a method
            node_type = "MethodDef"
            full_name = f"{self.current_class_name}.{node.name}"
            logging.debug(f"Found Method: {full_name}")
        else:
            # This is a top-level function
            node_type = "FunctionDef"
            full_name = node.name
            logging.debug(f"Found Function: {full_name}")

        self.nodes.append({
            "type": node_type,
            "name": full_name,
            "node_object": node # We keep the object for potential future steps
        })
        # **Crucial**: We stop traversing here.
        # We don't care about 'if' statements or 'for' loops *inside*
        # the function. The function itself is the "chunk".
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

    Input: { file_path: full_ast_object }
    Output: { file_path: [ list_of_key_node_dicts ] }
    """
    
    hierarchical_node_registry = {}
    
    logging.info(f"Extracting key nodes from {len(ast_registry)} ASTs...")
    
    for file_path, ast_obj in ast_registry.items():
        if not ast_obj:
            continue
            
        logging.debug(f"Visiting AST for: {file_path}")
        
        # 1. Create a new visitor instance for each file
        visitor = HierarchicalNodeVisitor()
        
        # 2. Traverse the full AST. The visitor will
        # automatically collect the nodes it cares about.
        visitor.visit(ast_obj)
        
        # 3. Store the simplified list of nodes
        hierarchical_node_registry[file_path] = visitor.nodes
        
        logging.debug(f"Extracted {len(visitor.nodes)} key nodes from: {file_path}")

    return hierarchical_node_registry


def get_ast_registry(start_directory):
    """
    Runs the full Task 1.1 and 1.2 pipeline.
    """
    # ... (this function is unchanged) ...
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


# --- [ NEW FUNCTION FOR JSON EXPORT ] ---
def export_registry_to_json(registry, output_filename):
    """
    Exports the hierarchical registry to a JSON file.
    It creates a copy of the registry without the non-serializable
    'node_object' items.
    """
    logging.info(f"\n--- [ Exporting Registry to {output_filename} ] ---")
    
    serializable_registry = {}
    
    for file_path, nodes in registry.items():
        serializable_nodes = []
        for node_info in nodes:
            # Create a copy and remove the AST object
            # which is not JSON-serializable
            serializable_info = node_info.copy()
            del serializable_info['node_object']
            serializable_nodes.append(serializable_info)
        
        # Use relative path for cleaner JSON
        relative_path = os.path.relpath(file_path)
        serializable_registry[relative_path] = serializable_nodes

    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_registry, f, indent=4)
        logging.info(f"Successfully exported node registry to {output_filename}")
    except (IOError, TypeError) as e:
        logging.error(f"Failed to export JSON file: {e}")
# --- [ END NEW FUNCTION ] ---


if __name__ == "__main__":
    # --- Main Execution ---
    
    target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"

    # --- Run Tasks 1.1 and 1.2 ---
    full_ast_registry = get_ast_registry(target_project_path)
    
    if not full_ast_registry:
        logging.info("Exiting.")
        exit()

    # --- Task 1.3: Run the Extractor ---
    logging.info("\n--- [ Task 1.3: Extracting Hierarchical Nodes ] ---")
    
    # This is the "simplified list of code chunks"
    hierarchical_node_registry = extract_hierarchical_nodes(full_ast_registry)

    # --- Summary & Debugging ---
    logging.info("\n--- [ Node Extraction Summary ] ---")
    # ... (summary logging is unchanged) ...
    logging.info(f"Processed {len(hierarchical_node_registry)} files.")
    
    total_nodes = 0
    for file_path, nodes in hierarchical_node_registry.items():
        total_nodes += len(nodes)
        
    logging.info(f"Total key nodes (chunks) extracted: {total_nodes}")

    # --- Detailed "Debug" Print ---
    if hierarchical_node_registry:
        logging.info("\n--- [ Detailed Hierarchical Node Registry ] ---")
        # ... (detailed logging is unchanged) ...
        for file_path, nodes in hierarchical_node_registry.items():
            short_path = os.path.basename(file_path)
            logging.info(f"\n--- [ {short_path} ] ---")
            if not nodes:
                logging.info("  (No classes or functions found)")
                continue
                
            for node_info in nodes:
                logging.info(f"  > Found {node_info['type']}: {node_info['name']}")

    # --- [ NEW: Export the result to JSON ] ---
    if hierarchical_node_registry:
        export_registry_to_json(hierarchical_node_registry, 'hierarchical_nodes.json')

