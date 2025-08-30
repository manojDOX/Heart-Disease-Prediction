# Import the pandas library for working with data
import pandas as pd
# Import the os library to handle file paths reliably
import os

def load_data(filename='heart.csv'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    # the 'data' folder, and the filename.
    path = os.path.join(project_root, 'data', filename)

    try:
        # Try to read the CSV file from the constructed absolute path
        df = pd.read_csv(path)

        # If successful, print a confirmation message
        print(f"✅ Data loaded successfully from: {path}")

        # Return the loaded DataFrame (table of data)
        return df

    # If the file is not found at the given path, show an error
    except FileNotFoundError:
        print(f"❌ File not found. The script looked for the file at this path:")
        print(f"   -> {path}")
        print("\nPlease double-check that your 'data' folder and 'heart.csv' file exist at that location.")
        return None