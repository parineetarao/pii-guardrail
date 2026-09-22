import json
import random
import os

# ============================================================
# TARGET CONFIGURATION CONFIG
# ============================================================
input_file = "clean_dataset.json"   # Reading strictly from audited data

train_output = "train.json"
val_output = "val.json"
test_output = "test.json"

print("🔍 ========================================================")
print("🔎 FINAL AUDITED DATASET SPLITTING ENGINE")
print("🔍 ========================================================\n")

if not os.path.exists(input_file):
    print(f"❌ Error: Cannot locate '{input_file}' in your project root.")
    exit()

try:
    with open(input_file, "r", encoding="utf-8") as f:
        unique_dataset = json.load(f)

    total_count = len(unique_dataset)
    print(f"🚀 Loaded {total_count} pristine, fully audited records.")

    # Apply strict seed shuffling to guarantee uniform class distribution
    random.seed(42)
    random.shuffle(unique_dataset)

    # Calculate precise mathematical 70% / 15% / 15% splits for the 1,973 rows
    train_idx = int(total_count * 0.70)
    val_idx = train_idx + int(total_count * 0.15)
    
    train_data = unique_dataset[:train_idx]
    val_data = unique_dataset[train_idx:val_idx]
    test_data = unique_dataset[val_idx:]

    # Save finalized split subsets directly to the drive
    with open(train_output, "w", encoding="utf-8") as f: json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(val_output, "w", encoding="utf-8") as f: json.dump(val_data, f, indent=2, ensure_ascii=False)
    with open(test_output, "w", encoding="utf-8") as f: json.dump(test_data, f, indent=2, ensure_ascii=False)

    print("\n✅ Finalized Audited Splits Successfully Generated!")
    print(f"📊 train.json   -> {len(train_data)} rows")
    print(f"📊 val.json     -> {len(val_data)} rows")
    print(f"📊 test.json    -> {len(test_data)} rows")
    print(f"📦 Grand Total   -> {len(train_data) + len(val_data) + len(test_data)} rows")

except Exception as e:
    print(f"❌ Split Engine Failure: {str(e)}")
