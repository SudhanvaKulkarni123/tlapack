import sys
import random

# Iterate over the paths in sys.path to find the directory containing the random module
random_module_path = None
for path in sys.path:
    if path.endswith('site-packages'):
        random_module_path = path + '/random.py'
        break

if random_module_path:
    print("Path to random module:", random_module_path)
else:
    print("Random module not found in sys.path")