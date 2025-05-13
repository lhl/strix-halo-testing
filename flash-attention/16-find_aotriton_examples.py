# find_aotriton_examples.py
import os
import glob

def find_python_files_with_aotriton():
    """Search for Python files that might show how to use AOTriton."""
    paths_to_search = [
        "/opt/rocm/",
        "/usr/local/lib/python*/site-packages/",
        "/home/",
    ]
    
    keywords = ['aotriton', 'attn_fwd', 'HipMemory', 'T4']
    
    for base_path in paths_to_search:
        for path in glob.glob(f"{base_path}**/*.py", recursive=True):
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    if any(keyword in content for keyword in keywords):
                        print(f"Found in {path}")
                        # Print relevant lines
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if any(keyword in line for keyword in keywords):
                                print(f"  Line {i+1}: {line.strip()}")
            except:
                pass

# Run search
print("=== Searching for AOTriton examples ===")
find_python_files_with_aotriton()
