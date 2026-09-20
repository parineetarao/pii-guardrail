import json
import random

# ============================================================
# CONFIG
# ============================================================

master_file_path = "dataset.json"
train_output_path = "train.json"
val_output_path = "val.json"

# ============================================================
# LOAD MASTER DATASET
# ============================================================

try:
    with open(master_file_path, "r", encoding="utf-8") as f:
        raw_content = f.read().strip()

    cleaned_pairs = []

    # ========================================================
    # STREAM JSON DECODER
    # Handles multiple JSON objects appended together
    # ========================================================

    decoder = json.JSONDecoder()
    pos = 0
    content_length = len(raw_content)
    raw_entries = []

    print("🔍 Activating stream decoder to parse stacked dataset layers...")

    while pos < content_length:

        # Skip whitespace/newlines
        while pos < content_length and raw_content[pos].isspace():
            pos += 1

        if pos >= content_length:
            break

        try:
            result, next_pos = decoder.raw_decode(raw_content, pos)

            # If the decoded object is a list, flatten it
            if isinstance(result, list):
                raw_entries.extend(result)
            else:
                raw_entries.append(result)

            pos = next_pos

        except json.JSONDecodeError:
            # Move forward one character and search for
            # the next valid JSON structure
            pos += 1

    print(
        f"📦 Successfully parsed {len(raw_entries)} raw structural "
        f"elements from stream."
    )

    # ========================================================
    # EXTRACTION DIAGNOSTICS
    # ========================================================

    stats = {
        "total": 0,
        "standard_gemini": 0,
        "n8n_gemini": 0,
        "direct_pair": 0,
        "direct_text": 0,
        "no_recognized_structure": 0,
        "empty_text": 0,
        "json_parse_error": 0,
        "not_dict": 0,
        "missing_fields": 0,
        "skipped_source": 0,
        "valid": 0
    }

    # ========================================================
    # EXTRACT source_text / masked_text
    # ========================================================

    print("🔎 Extracting source_text and masked_text...")

    for entry in raw_entries:

        stats["total"] += 1

        if not entry:
            continue

        try:
            text_content = None

            # ------------------------------------------------
            # CASE 1:
            # Standard Gemini API structure
            #
            # candidates
            #   └── [0]
            #       └── content
            #           └── parts
            #               └── [0]
            #                   └── text
            # ------------------------------------------------

            if isinstance(entry, dict) and "candidates" in entry:

                candidates = entry["candidates"]

                if (
                    isinstance(candidates, list)
                    and len(candidates) > 0
                ):
                    content = candidates[0].get("content", {})

                    if isinstance(content, dict):

                        parts = content.get("parts", [])

                        if (
                            isinstance(parts, list)
                            and len(parts) > 0
                        ):
                            text_content = parts[0].get("text", "")

                            stats["standard_gemini"] += 1

            # ------------------------------------------------
            # CASE 2:
            # n8n wrapper around Gemini response
            #
            # json
            #   └── candidates
            #       └── [0]
            # ------------------------------------------------

            elif (
                isinstance(entry, dict)
                and "json" in entry
                and isinstance(entry["json"], dict)
                and "candidates" in entry["json"]
            ):

                candidates = entry["json"]["candidates"]

                if (
                    isinstance(candidates, list)
                    and len(candidates) > 0
                ):
                    content = candidates[0].get("content", {})

                    if isinstance(content, dict):

                        parts = content.get("parts", [])

                        if (
                            isinstance(parts, list)
                            and len(parts) > 0
                        ):
                            text_content = parts[0].get("text", "")

                            stats["n8n_gemini"] += 1

            # ------------------------------------------------
            # CASE 3:
            # Already flattened source/masked pair
            # ------------------------------------------------

            elif (
                isinstance(entry, dict)
                and "source_text" in entry
                and "masked_text" in entry
            ):

                src = entry.get("source_text", "")
                msk = entry.get("masked_text", "")

                if src and msk:

                    cleaned_pairs.append({
                        "source_text": src.strip(),
                        "masked_text": msk.strip()
                    })

                    stats["direct_pair"] += 1
                    stats["valid"] += 1

                continue

            # ------------------------------------------------
            # CASE 4:
            # n8n direct text wrapper
            # ------------------------------------------------

            elif (
                isinstance(entry, dict)
                and "json" in entry
                and isinstance(entry["json"], dict)
                and "text" in entry["json"]
            ):

                text_content = entry["json"]["text"]
                stats["direct_text"] += 1

            # ------------------------------------------------
            # No recognized structure
            # ------------------------------------------------

            else:
                stats["no_recognized_structure"] += 1
                continue

            # ------------------------------------------------
            # No text extracted
            # ------------------------------------------------

            if not text_content:
                stats["empty_text"] += 1
                continue

            # ------------------------------------------------
            # Gemini returns the actual pair as a JSON STRING
            # ------------------------------------------------

            if isinstance(text_content, str):

                clean_string = (
                    text_content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                try:
                    parsed_pair = json.loads(clean_string)

                except json.JSONDecodeError:
                    stats["json_parse_error"] += 1
                    continue

            else:
                parsed_pair = text_content

            # ------------------------------------------------
            # Make sure parsed result is a dictionary
            # ------------------------------------------------

            if not isinstance(parsed_pair, dict):
                stats["not_dict"] += 1
                continue

            # ------------------------------------------------
            # Extract required fields
            # ------------------------------------------------

            src = parsed_pair.get("source_text", "")
            msk = parsed_pair.get("masked_text", "")

            if not src or not msk:
                stats["missing_fields"] += 1
                continue

            # ------------------------------------------------
            # Remove known bad pipeline outputs
            # ------------------------------------------------

            if "Skipped" in src:
                stats["skipped_source"] += 1
                continue

            if "Parsing Exception" in msk:
                stats["skipped_source"] += 1
                continue

            # ------------------------------------------------
            # VALID TRAINING PAIR
            # ------------------------------------------------

            cleaned_pairs.append({
                "source_text": src.strip(),
                "masked_text": msk.strip()
            })

            stats["valid"] += 1

        except Exception as e:

            # Unexpected structural error.
            # We intentionally continue so one bad record
            # cannot kill the entire dataset extraction.
            continue

    # ========================================================
    # PRINT DIAGNOSTICS
    # ========================================================

    print("\n========== EXTRACTION DIAGNOSTICS ==========")

    print(f"Total raw entries             : {stats['total']}")
    print(f"Standard Gemini responses     : {stats['standard_gemini']}")
    print(f"n8n Gemini responses          : {stats['n8n_gemini']}")
    print(f"Direct source/masked pairs    : {stats['direct_pair']}")
    print(f"Direct text wrappers          : {stats['direct_text']}")
    print(f"No recognized structure       : {stats['no_recognized_structure']}")
    print(f"Empty text                    : {stats['empty_text']}")
    print(f"JSON parse errors             : {stats['json_parse_error']}")
    print(f"Non-dictionary payloads       : {stats['not_dict']}")
    print(f"Missing source/masked fields  : {stats['missing_fields']}")
    print(f"Skipped/bad outputs           : {stats['skipped_source']}")
    print(f"VALID TRAINING PAIRS          : {stats['valid']}")

    print("============================================\n")

    # ========================================================
    # SAFETY CHECK
    # ========================================================

    if len(cleaned_pairs) == 0:

        print(
            "❌ ERROR: No valid training pairs were extracted."
        )

        exit()

    # ========================================================
    # SHUFFLE
    # ========================================================

    random.seed(42)
    random.shuffle(cleaned_pairs)

    # ========================================================
    # 80/20 TRAIN / VALIDATION SPLIT
    # ========================================================

    split_index = int(len(cleaned_pairs) * 0.8)

    train_data = cleaned_pairs[:split_index]
    val_data = cleaned_pairs[split_index:]

    # ========================================================
    # SAVE TRAIN
    # ========================================================

    with open(train_output_path, "w", encoding="utf-8") as f:

        json.dump(
            train_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # SAVE VALIDATION
    # ========================================================

    with open(val_output_path, "w", encoding="utf-8") as f:

        json.dump(
            val_data,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("✅ Train-Validation Split Complete!")

    print(
        f"📊 Total valid records : {len(cleaned_pairs)}"
    )

    print(
        f"📊 Training records    : {len(train_data)}"
    )

    print(
        f"📊 Validation records  : {len(val_data)}"
    )

    print(
        f"💾 Saved training data : {train_output_path}"
    )

    print(
        f"💾 Saved validation    : {val_output_path}"
    )

except Exception as e:

    print(
        f"❌ Critical failure: {str(e)}"
    )