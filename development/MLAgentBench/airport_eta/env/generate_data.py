import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DROP_COLUMNS = {
    "TODAY_ID",
    "S_DEPAP",
    "S_ARRAP",
    "P_DEPAP",
    "P_ARRAP",
    "R_ARRTIME",
    "callsign",
    "track_ids",
    "points_num",
    "last_timestamp",
    "last_lon",
    "last_lat",
    "last_height",
    "Delay_Time",
    "Actual_Flight_Time",
    "R_ARRAP",
    "cluster_open",
}


def build_split(source_name, output_name):
    source_path = ROOT / source_name
    output_path = ROOT / output_name

    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = [name for name in reader.fieldnames if name not in DROP_COLUMNS]
        rows = []

        for row in reader:
            # Keep only flights whose scheduled and realized arrival airport is ZBAA.
            if row["R_ARRAP"] != "ZBAA" or row["S_ARRAP"] != "ZBAA":
                continue
            rows.append({name: row[name] for name in fieldnames})

    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_name}.")


if __name__ == "__main__":
    build_split("flight_entry_events_with_transfer_intent_202508_ZBAA.csv", "train.csv")
    build_split("flight_entry_events_with_transfer_intent_202509_ZBAA.csv", "test.csv")
