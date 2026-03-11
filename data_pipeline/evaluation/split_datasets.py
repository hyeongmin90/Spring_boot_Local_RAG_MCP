import json
import random

def split_datasets(file1, output1, output2, ratio=0.3):
    # Load first dataset
    with open(file1, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"Loaded {len(data)} items from {file1}")
    
    # deduplicate by id
    unique_items = {}
    
    for item in data:
        item_id = item.get("id")
        if item_id and item_id not in unique_items:
            unique_items[item_id] = item
            
    final_list = list(unique_items.values())
    print(f"Total unique items after deduplication: {len(final_list)}")
    
    # Shuffle for random split
    random.shuffle(final_list)
    
    # Split
    split_index = int(len(final_list) * ratio)
    part1 = final_list[:split_index]
    part2 = final_list[split_index:]
    
    print(f"Split {ratio*100:.0f}:{(1-ratio)*100:.0f} -> {len(part1)} items and {len(part2)} items")
    
    # Save output 1
    with open(output1, 'w', encoding='utf-8') as f:
        json.dump(part1, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(part1)} items to {output1}")
    
    # Save output 2
    with open(output2, 'w', encoding='utf-8') as f:
        json.dump(part2, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(part2)} items to {output2}")

if __name__ == "__main__":
    import os
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file1 = os.path.join(base_dir, "evaluation_dataset.json")
    
    output1 = os.path.join(base_dir, "test_v1.json")
    output2 = os.path.join(base_dir, "dev_v1.json")
    
    split_datasets(file1, output1, output2, ratio=0.3)
