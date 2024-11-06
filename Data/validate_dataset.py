import json

# Load the JSON dataset for validation
with open('data/final_global_context_dataset.json', 'r') as f:
    global_context_dataset = json.load(f)

# Check a few cycles to verify structure
print("Total cycles:", len(global_context_dataset))
print("Sample cycle event sequence:", global_context_dataset[0])

# Additional checks
for i, cycle in enumerate(global_context_dataset):
    assert isinstance(cycle, list), f"Cycle {i} is not a list."
    print(f"Cycle {i} - Event Count: {len(cycle)} - Events: {cycle}")
