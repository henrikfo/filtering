#!/bin/bash

# Check if start and end date are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_date> <end_date>"
    exit 1
fi

start_date="$1"
end_date="$2"

# Convert start and end dates to seconds since epoch
start_sec=$(date -d "$start_date" +%s)
end_sec=$(date -d "$end_date" +%s)

# Loop through the date range
current_sec=$start_sec
while [ $current_sec -le $end_sec ]; do
    # Convert current date to the desired format
    current_date=$(date -d "@$current_sec" +%Y-%m-%d)

    # Use sed to update the date in the filtering.json file
    sed -i "s/\"date\": \"[^\"]*\"/\"date\": \"$current_date\"/" filtering.json

    # Run the command with the updated filtering.json
    colonies function submit --spec filtering.json

    # Move to the next day
    current_sec=$((current_sec + 86400))  # Add 86400 seconds (1 day)
done

