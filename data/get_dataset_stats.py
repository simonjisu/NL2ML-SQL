import json
from pathlib import Path
from collections import defaultdict

def count_data(path: Path):
    counter = defaultdict(set)
    with open(path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    for d in data:
        counter['schema'].add(json.dumps(d['schema'], indent=2))
        
        if d['intent'].get('update_condition'):
            counter['s2'].add(json.dumps(d['intent'], indent=2))
            counter['s2q'].add(d['question'])
        else:
            counter['s1'].add(json.dumps(d['intent'], indent=2))
            counter['s1q'].add(d['question'])

        counter['intent'].add(json.dumps(d['intent'], indent=2))
        counter['question'].add(d['question'])
    return counter

if __name__ == "__main__":
    for name, path in zip(['SEED', 'TRAIN', 'TEST'], 
                          ['./train_seed_cots.jsonl', './train_dataset.jsonl', './test_dataset.jsonl']):
        counter = count_data(Path(path))
        print(f"Statistics for {name}:")
        print(f"Total unique schemas: {len(counter['schema'])}")
        print(f"Total unique intent: {len(counter['intent'])} | S1: {len(counter['s1'])} | S2: {len(counter['s2'])}")
        print(f"Total unique questions: {len(counter['question'])} | S1: {len(counter['s1q'])} | S2: {len(counter['s2q'])}")
        print()