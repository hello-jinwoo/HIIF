"""
Validate JSON preference files for potential bugs.
Checks for cases where prefer_idx == non_prefer_idx.
"""

import json
import os
import glob
from collections import defaultdict


def validate_json_files(responses_dir='./load/PPS/responses'):
    """
    Check all JSON files in the responses directory for bugs.

    Args:
        responses_dir: Path to the directory containing JSON response files
    """
    print("=" * 80)
    print("PPS Preference JSON Validation")
    print("=" * 80)

    # Find all JSON files
    json_pattern = os.path.join(responses_dir, '**/*.json')
    json_files = glob.glob(json_pattern, recursive=True)

    print(f"\nFound {len(json_files)} JSON files to check\n")

    # Statistics
    total_pairs = 0
    bug_count = 0
    bug_details = []

    # Check each file
    for json_file in sorted(json_files):
        user_id = os.path.basename(json_file).replace('.json', '')

        try:
            with open(json_file, 'r') as f:
                responses = json.load(f)

            for image_id, prefs in responses.items():
                total_pairs += 1
                prefer_idx = prefs.get('prefer', None)
                non_prefer_idx = prefs.get('non_prefer', None)

                # Check for bug: prefer_idx == non_prefer_idx
                if prefer_idx == non_prefer_idx:
                    bug_count += 1
                    bug_details.append({
                        'user_id': user_id,
                        'image_id': image_id,
                        'prefer_idx': prefer_idx,
                        'non_prefer_idx': non_prefer_idx,
                        'json_file': json_file
                    })

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    # Print results
    print("-" * 80)
    print("RESULTS")
    print("-" * 80)
    print(f"Total preference pairs checked: {total_pairs}")
    print(f"Buggy pairs (prefer_idx == non_prefer_idx): {bug_count}")
    print(f"Bug rate: {bug_count / total_pairs * 100:.2f}%")
    print()

    if bug_count > 0:
        print("=" * 80)
        print("BUG DETAILS")
        print("=" * 80)
        print(f"\nFound {bug_count} cases where prefer_idx == non_prefer_idx:")
        print("This means BOTH paths point to the SAME IMAGE FILE!\n")

        # Show first 20 bugs
        for i, bug in enumerate(bug_details[:20], 1):
            print(f"{i}. User: {bug['user_id']}, Image: {bug['image_id']}")
            print(f"   prefer_idx = non_prefer_idx = {bug['prefer_idx']}")
            print(f"   File: {bug['json_file']}")
            print()

        if len(bug_details) > 20:
            print(f"... and {len(bug_details) - 20} more bugs")
            print()

        # Save full bug report
        report_file = './bug_report_json_preferences.txt'
        with open(report_file, 'w') as f:
            f.write("PPS Preference JSON Bug Report\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total pairs: {total_pairs}\n")
            f.write(f"Buggy pairs: {bug_count}\n")
            f.write(f"Bug rate: {bug_count / total_pairs * 100:.2f}%\n\n")
            f.write("Full list of buggy pairs:\n")
            f.write("-" * 80 + "\n")
            for bug in bug_details:
                f.write(f"User: {bug['user_id']}, Image: {bug['image_id']}, ")
                f.write(f"idx: {bug['prefer_idx']}, File: {bug['json_file']}\n")

        print(f"Full bug report saved to: {report_file}")
        print()
    else:
        print("✓ No bugs found! All prefer/non-prefer pairs use different indices.")
        print()

    print("=" * 80)

    return bug_count, total_pairs, bug_details


if __name__ == '__main__':
    import sys

    # Allow custom path
    responses_dir = './load/PPS/responses'
    if len(sys.argv) > 1:
        responses_dir = sys.argv[1]

    bug_count, total_pairs, bug_details = validate_json_files(responses_dir)

    # Exit with error code if bugs found
    sys.exit(1 if bug_count > 0 else 0)
