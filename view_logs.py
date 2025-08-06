#!/usr/bin/env python3
"""
Simple script to view and manage API logs
"""

import os
import glob
from datetime import datetime

def list_log_files():
    """List all available log files."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print("No logs directory found.")
        return []
    
    log_files = glob.glob(os.path.join(logs_dir, "api_log_*.txt"))
    return sorted(log_files, reverse=True)  # Most recent first

def view_latest_log():
    """View the most recent log file."""
    log_files = list_log_files()
    if not log_files:
        print("No log files found.")
        return
    
    latest_log = log_files[0]
    print(f"📄 Viewing latest log: {os.path.basename(latest_log)}")
    print("=" * 80)
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Error reading log file: {e}")

def view_specific_log(filename):
    """View a specific log file."""
    logs_dir = "logs"
    log_path = os.path.join(logs_dir, filename)
    
    if not os.path.exists(log_path):
        print(f"Log file not found: {filename}")
        return
    
    print(f"📄 Viewing log: {filename}")
    print("=" * 80)
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
    except Exception as e:
        print(f"Error reading log file: {e}")

def main():
    """Main function to handle log viewing."""
    print("🔍 API Log Viewer")
    print("=" * 50)
    
    log_files = list_log_files()
    
    if not log_files:
        print("No log files found. Make some API requests first!")
        return
    
    print(f"Found {len(log_files)} log file(s):")
    for i, log_file in enumerate(log_files[:10]):  # Show only last 10
        filename = os.path.basename(log_file)
        file_size = os.path.getsize(log_file)
        file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
        print(f"  {i+1}. {filename} ({file_size} bytes, {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
    
    if len(log_files) > 10:
        print(f"  ... and {len(log_files) - 10} more files")
    
    print("\nOptions:")
    print("  1. View latest log")
    print("  2. View specific log file")
    print("  3. List all log files")
    print("  4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        view_latest_log()
    elif choice == "2":
        filename = input("Enter log filename (e.g., api_log_20241201_143022.txt): ").strip()
        view_specific_log(filename)
    elif choice == "3":
        print("\nAll log files:")
        for i, log_file in enumerate(log_files):
            filename = os.path.basename(log_file)
            file_size = os.path.getsize(log_file)
            file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
            print(f"  {i+1}. {filename} ({file_size} bytes, {file_time.strftime('%Y-%m-%d %H:%M:%S')})")
    elif choice == "4":
        print("Goodbye!")
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main() 