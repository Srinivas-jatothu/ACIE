import os
import logging
from datetime import datetime

# --- Configuration ---

# 1. Define supported programming language extensions
# You can add or remove extensions as needed
SUPPORTED_EXTENSIONS = {
    '.py',    # Python
    '.java',  # Java
    '.ts',    # TypeScript
    '.js',    # JavaScript
    '.cpp',   # C++
    '.c',     # C
    '.h',     # C/C++ Header
    '.cs',    # C#
    '.go',    # Go
    '.rs',    # Rust
    '.swift', # Swift
    '.kt',    # Kotlin
}

# 2. Set up logging for debugging
# This will print messages to your console to show what the script is doing
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# --- End Configuration ---


def scan_codebase(start_directory):
    """
    Recursively scans a directory for source code files and gathers metadata.

    Args:
        start_directory (str): The absolute path to the project's root directory.

    Returns:
        list: A list of dictionaries, where each dictionary is a "file metadata record".
              Returns an empty list if the directory is invalid.
    """
    
    # 3. Establish the file metadata registry
    file_metadata_registry = []
    
    if not os.path.isdir(start_directory):
        logging.error(f"Directory does not exist: {start_directory}")
        return file_metadata_registry

    logging.info(f"Starting recursive scan of: {start_directory}")

    # 4. Create the file scanning mechanism (recursive traversal)
    for root_dir, sub_dirs, files in os.walk(start_directory):
        logging.debug(f"Scanning directory: {root_dir}")
        
        for file in files:
            # 5. Filter files based on supported extensions
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in SUPPORTED_EXTENSIONS:
                try:
                    # Get full file path
                    file_path = os.path.join(root_dir, file)
                    
                    # 6. Track file metadata (path, size, modified time)
                    stats = os.stat(file_path)
                    
                    file_size_bytes = stats.st_size
                    last_modified_timestamp = stats.st_mtime
                    
                    # Convert timestamp to a readable string
                    last_modified_human = datetime.fromtimestamp(
                        last_modified_timestamp
                    ).isoformat()

                    # Create the metadata record
                    file_record = {
                        "path": file_path,
                        "size_bytes": file_size_bytes,
                        "last_modified": last_modified_human
                    }
                    
                    # Add record to the registry
                    file_metadata_registry.append(file_record)
                    logging.debug(f"Found source file: {file_path}")

                except (IOError, FileNotFoundError, PermissionError) as e:
                    logging.warning(f"Could not access or stat file {file_path}: {e}")
            
            else:
                logging.debug(f"Skipping non-source file: {file}")

    logging.info(f"Scan complete. Found {len(file_metadata_registry)} source files.")
    return file_metadata_registry


if __name__ == "__main__":
    # --- Main Execution ---
    
    # Set the target directory you mentioned
    # IMPORTANT: Python needs double backslashes '\' or a 'raw' string (r"...")
    # for Windows paths.
    target_project_path = r"C:\Users\jsrin\OneDrive\Desktop\ACIE\Testcase\Shopping"

    # --- OR, to make it interactive (uncomment the 3 lines below) ---
    # print("Enter the full path to your project directory:")
    # target_project_path = input(r"> ")
    # -----------------------------------------------------------------

    # Run the scanner
    registry = scan_codebase(target_project_path)
    
    # Print the final registry
    if registry:
        print("\n--- [ File Metadata Registry ] ---")
        for i, record in enumerate(registry):
            print(f"\nRecord {i + 1}:")
            print(f"  Path:     {record['path']}")
            print(f"  Size:     {record['size_bytes']} bytes")
            print(f"  Modified: {record['last_modified']}")
        print("\n--- [ End of Registry ] ---")
    else:
        print(f"\nNo source files found in {target_project_path} or directory was not valid.")
