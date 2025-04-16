# check_flower_version.py
import flwr as fl
import pkg_resources
import sys

def check_flower_version():
    # Get the installed version of Flower
    version = pkg_resources.get_distribution("flwr").version
    print(f"Installed Flower version: {version}")
    
    # Check if ndarrays_to_parameters exists
    has_ndarrays_to_parameters = hasattr(fl.common, "ndarrays_to_parameters")
    print(f"Has fl.common.ndarrays_to_parameters: {has_ndarrays_to_parameters}")
    
    # Check if parameters_to_ndarrays exists
    has_parameters_to_ndarrays = hasattr(fl.common, "parameters_to_ndarrays")
    print(f"Has fl.common.parameters_to_ndarrays: {has_parameters_to_ndarrays}")
    
    # Check if weights_to_parameters exists
    has_weights_to_parameters = hasattr(fl.common, "weights_to_parameters")
    print(f"Has fl.common.weights_to_parameters: {has_weights_to_parameters}")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check module path
    print(f"Flower module path: {fl.__file__}")
    
    # Print recommended updates based on version
    if has_ndarrays_to_parameters:
        print("\nRecommendation: Use ndarrays_to_parameters and parameters_to_ndarrays in your code")
    else:
        print("\nRecommendation: Use weights_to_parameters and parameters_to_weights in your code")

if __name__ == "__main__":
    check_flower_version()