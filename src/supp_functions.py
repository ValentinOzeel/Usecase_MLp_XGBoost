import importlib.util

def import_module_from_path(module_path):
    """
    Dynamically imports a Python module from its path.

    Args:
    - module_path (str): The path to the Python module.

    Returns:
    - module: The imported module.
    """
    # Create a spec object representing the module
    spec = importlib.util.spec_from_file_location("", module_path)
    
    # Load the module based on the spec
    module = importlib.util.module_from_spec(spec)
    
    # Finalize the loading process
    spec.loader.exec_module(module)
    
    return module