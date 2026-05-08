import json
import base64
import os
from pathlib import Path

def extract_notebook_data(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results = {
        "train_logs": [],
        "reward_gap": None,
        "final_loss": None,
        "side_by_side": [],
        "benchmarks": {},
        "images": {}
    }
    
    for cell in nb['cells']:
        # Extract logs from outputs
        if cell['cell_type'] == 'code':
            for output in cell.get('outputs', []):
                if 'text' in output:
                    text = "".join(output['text'])
                    results["train_logs"].append(text)
                
                # Extract images
                if 'data' in output and 'image/png' in output['data']:
                    # We can't easily map them to names without context, but we can look for cell metadata or previous markdown
                    img_data = output['data']['image/png']
                    # Use a simple counter or heuristic
                    img_id = len(results["images"])
                    results["images"][img_id] = img_data

    return results

repo_root = Path("d:/AiGit/day22/Day22-Track3-DPO-Alignment-Lab")
nb_path = repo_root / "colab" / "Lab22_DPO_T4.ipynb"
data = extract_notebook_data(nb_path)

# Write extracted logs to a text file for review
with open(repo_root / "scripts" / "extracted_logs.txt", "w", encoding="utf-8") as f:
    for log in data["train_logs"]:
        f.write(log + "\n" + "="*80 + "\n")

# Extract images to screenshots folder
screenshot_dir = repo_root / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)

for i, b64 in data["images"].items():
    with open(screenshot_dir / f"extracted_image_{i}.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"Extracted image {i}")
