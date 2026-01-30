import os
from glob import glob

# Folder path for logs
log_folder_path = "logs/"

# Function to delete logs for a specific date
def delete_logs_for_date(date):
    # Get all log files that match the specified date
    log_files = glob(os.path.join(log_folder_path, f"*{date}*.log"))

    # Check if any files were found
    if log_files:
        for log_file_path in log_files:
            try:
                # Delete the log file
                os.remove(log_file_path)
                print(f"Deleted log file: {log_file_path}")
            except Exception as e:
                print(f"Error deleting file {log_file_path}: {e}")
    else:
        print(f"No log files found for date: {date}")

# Example usage: Delete logs for a specific date (e.g., "2024_10_09")
delete_logs_for_date("2025_02_18")
