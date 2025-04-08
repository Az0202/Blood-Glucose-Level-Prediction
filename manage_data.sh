#!/bin/bash
# manage_data.sh
# Helper script for managing the OhioT1DM dataset

# Print usage information
print_usage() {
    echo "OhioT1DM Dataset Management Script"
    echo ""
    echo "Usage: ./manage_data.sh [command]"
    echo ""
    echo "Commands:"
    echo "  create-sample   Generate synthetic sample data"
    echo "  clean           Remove all data files"
    echo "  clean-keep      Remove real data but keep sample data"
    echo "  backup          Create a backup of the data directory"
    echo "  restore         Restore from backup"
    echo "  help            Display this help message"
    echo ""
}

# Check for the command line argument
if [ $# -eq 0 ]; then
    print_usage
    exit 1
fi

# Process the command
case "$1" in
    create-sample)
        echo "Generating synthetic sample data..."
        python create_sample_data.py
        ;;
    
    clean)
        echo "Removing all data files..."
        python clean_data.py
        ;;
    
    clean-keep)
        echo "Removing real data but keeping sample data..."
        python clean_data.py --keep_sample
        ;;
    
    backup)
        echo "Creating a backup of the data directory..."
        python clean_data.py --backup
        ;;
    
    restore)
        if [ -d "DATA_backup" ]; then
            echo "Restoring data from backup..."
            rm -rf DATA
            cp -r DATA_backup DATA
            echo "Data restored from backup."
        else
            echo "Error: No backup directory found at DATA_backup"
            exit 1
        fi
        ;;
    
    help)
        print_usage
        ;;
    
    *)
        echo "Error: Unknown command '$1'"
        print_usage
        exit 1
        ;;
esac

exit 0 