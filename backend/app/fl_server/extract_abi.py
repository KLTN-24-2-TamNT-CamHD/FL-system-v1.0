#!/usr/bin/env python3
"""
Extract ABI from Solidity contract file.
This script requires solc (Solidity compiler) to be installed.
"""

import json
import subprocess
import sys
import os
from pathlib import Path

def extract_abi(contract_file_path, output_file=None):
    """
    Extract ABI from Solidity contract file using solc.
    
    Args:
        contract_file_path: Path to the Solidity contract file
        output_file: Path to save the ABI JSON (optional)
        
    Returns:
        ABI as a list
    """
    contract_file = Path(contract_file_path)
    
    if not contract_file.exists():
        print(f"Error: Contract file not found: {contract_file_path}")
        return None
    
    try:
        # Run solc to get the ABI
        cmd = [
            'solc',
            '--abi',
            str(contract_file)
        ]
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Extract ABI from output (the output contains the contract name and the ABI)
        output_lines = process.stdout.strip().split('\n')
        
        # Find the line after "======= <contract> =======", which should be the ABI
        contract_name = None
        abi_json = None
        
        for i, line in enumerate(output_lines):
            if line.startswith('======= ') and ' =======') in line:
                contract_name = line.split(':')[-1].split(' =====')[0].strip()
                # The ABI should be in the next line
                if i + 1 < len(output_lines):
                    abi_json = output_lines[i + 1]
                    break
        
        if not abi_json:
            print("Error: Could not find ABI in compiler output")
            return None
        
        # Parse the ABI
        abi = json.loads(abi_json)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(abi, f, indent=2)
            print(f"ABI saved to {output_file}")
        
        return abi
    
    except subprocess.CalledProcessError as e:
        print(f"Error running solc: {e}")
        print(f"STDERR: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing ABI JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_abi.py <contract_file.sol> [output_file.json]")
        sys.exit(1)
    
    contract_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Default output filename based on input
        output_file = Path(contract_file).stem + "_abi.json"
    
    abi = extract_abi(contract_file, output_file)
    if not abi:
        sys.exit(1)

if __name__ == "__main__":
    main()