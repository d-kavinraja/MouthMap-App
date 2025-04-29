import subprocess
import sys
import os

# Function to check if a package is installed
def is_package_installed(package):
    try:
        __import__(package)
        return True
    except ImportError:
        return False

# Function to install missing packages from requirements.txt
def install_requirements():
    if not os.path.exists("requirements.txt"):
        print("requirements.txt file not found!")
        sys.exit(1)

    print("Installing missing packages from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

# List of required packages (can be fetched from the 'requirements.txt' or listed manually)
required_packages = []

# Read the requirements.txt file and collect all package names
with open('requirements.txt', 'r') as f:
    required_packages = [line.strip() for line in f.readlines()]

# Check if all required packages are installed
missing_packages = []
for package in required_packages:
    if not is_package_installed(package):
        missing_packages.append(package)

# If any package is missing, install them
if missing_packages:
    print(f"The following packages are missing: {', '.join(missing_packages)}")
    install_requirements()
else:
    print("All required packages are already installed.")

