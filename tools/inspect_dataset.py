import argparse
import json
from collections import Counter
from pathlib import Path


def inspect_dataset(path: Path, limit: int = 1000):
    total = 0
    bad_json = 0
    missing_text = 0
    empty_text = 0
    lengths = []
    newline_counts = []
    assistant_hits = 0
    user_hits = 0
    im_start_hits = 0
    samples = []
    char_counter = Counter()

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            if total >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            total += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                continue

            text = obj.get("text")
            if text is None:
                missing_text += 1
                continue
            text = str(text)
            if not text.strip():
                empty_text += 1
                continue

            lengths.append(len(text))
            newline_counts.append(text.count("\n"))
            assistant_hits += text.count("Assistant:")
            user_hits += text.count("User:")
            im_start_hits += text.count("<|im_start|>")
            char_counter.update(text)
            if len(samples) < 3:
                samples.append((line_no, text[:400]))

    print(f"path: {path}")
    print(f"inspected_samples: {total}")
    print(f"bad_json: {bad_json}")
    print(f"missing_text: {missing_text}")
    print(f"empty_text: {empty_text}")

    if lengths:
        lengths_sorted = sorted(lengths)
        n = len(lengths_sorted)
        p50 = lengths_sorted[n // 2]
        p90 = lengths_sorted[min(n - 1, int(n * 0.9))]
        p99 = lengths_sorted[min(n - 1, int(n * 0.99))]
        avg_len = sum(lengths_sorted) / n
        avg_newlines = sum(newline_counts) / n
        print(f"avg_chars: {avg_len:.1f}")
        print(f"p50_chars: {p50}")
        print(f"p90_chars: {p90}")
        print(f"p99_chars: {p99}")
        print(f"avg_newlines: {avg_newlines:.1f}")

    print(f'user_marker_hits: {user_hits}')
    print(f'assistant_marker_hits: {assistant_hits}')
    print(f'im_start_hits: {im_start_hits}')

    common_chars = "".join(ch for ch, _ in char_counter.most_common(30))
    print(f"top_chars_preview: {common_chars}")

    print("\nexamples:")
    for line_no, sample in samples:
        print(f"[line {line_no}] {sample}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Inspect jsonl text dataset quality.")
    parser.add_argument(
        "--data_path", default="dataset/pretrain_hq.jsonl", type=str
    )
    parser.add_argument("--limit", default=1000, type=int)
    args = parser.parse_args()
    inspect_dataset(Path(args.data_path), args.limit)


if __name__ == "__main__":
    main()
