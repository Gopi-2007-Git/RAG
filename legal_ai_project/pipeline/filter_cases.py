import json
import os

METADATA_FOLDER = "data/metadata"

usable_decisions = {
    "appeal_allowed",
    "appeal_dismissed",
    "appeal_partly_allowed",
    "petition_allowed",
    "petition_dismissed",
    "petition_partly_allowed",
    "claim_allowed",
    "claim_dismissed",
    "application_allowed",
    "application_dismissed",
    "order_set_aside",
    "order_modified",
    "order_confirmed"
}

usable_cases = []
excluded_cases = []

for filename in os.listdir(METADATA_FOLDER):

    if not filename.endswith(".json"):
        continue

    case_name = filename.replace(".json", "")

    with open(
        os.path.join(METADATA_FOLDER, filename),
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)

    decision = metadata.get("decision", "not_identified")

    if decision in usable_decisions:
        usable_cases.append(case_name)
    else:
        excluded_cases.append(case_name)

print("\n===== CASE FILTER RESULTS =====")

print("\nUsable cases:")
for case in usable_cases:
    print("✓", case)

print("\nExcluded cases:")
for case in excluded_cases:
    print("✗", case)

print("\nTotal usable:", len(usable_cases))
print("Total excluded:", len(excluded_cases))