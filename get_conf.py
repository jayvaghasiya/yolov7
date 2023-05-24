import json

# Load the JSON data from file
with open('runs/test/exp7/best_predictions.json', 'r') as f:
    data = json.load(f)

# Create a dictionary to store the minimum confidence threshold for each class
class_thresholds = {}

# Iterate over the objects in the JSON data
for obj in data:
    category_id = obj['category_id']
    score = obj['score']

    # Check if the category_id is already present in the class_thresholds dictionary
    if category_id in class_thresholds:
        # Update the minimum confidence threshold for the class if necessary
        if score < class_thresholds[category_id]:
            class_thresholds[category_id] = score
    else:
        # Set the initial confidence threshold for the class
        class_thresholds[category_id] = score

# Print the minimum confidence thresholds for each class
for category_id, threshold in class_thresholds.items():
    print(f"Category {category_id}: Minimum confidence threshold = {threshold}")
