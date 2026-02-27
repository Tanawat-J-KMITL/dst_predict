import numpy as np
from datetime import datetime, timezone, timedelta

def parse_dst_line(data: str):
    """
    Parse a single IAGA-2002 Dst hourly line.
    Returns a list with one record dict, or empty list if header/comment.
    """

    line = data.strip()

    # Skip headers, comments, blank lines

    if line.endswith("|"):
        return "header"

    parts = line.split()

    # Expected format:
    # DATE TIME DOY DST
    if len(parts) < 4:
        return "invalid"

    date_str = parts[0]
    time_str = parts[1]
    dst_str = parts[3]

    # Combine date + time and parse as UTC
    return {
        "timestamp": datetime.strptime(
            f"{date_str} {time_str}",
            "%Y-%m-%d %H:%M:%S.%f"
        ).replace(tzinfo=timezone.utc),
        "dst_nT": float(dst_str)
    }

def hader_parser(header_data, line):
    if line.startswith("Format"):
        header_data["format"] = line.split("Format")[-1].rstrip("|").strip()

    elif line.startswith("Source of Data"):
        header_data["source"] = line.split("Source of Data")[-1].rstrip("|").strip()

    elif line.startswith("Station Name"):
        header_data["station"] = line.split("Station Name")[-1].rstrip("|").strip()

    elif line.startswith("IAGA CODE"):
        header_data["iaga_code"] = line.split("IAGA CODE")[-1].rstrip("|").strip()

    elif line.startswith("Data Interval Type"):
        header_data["interval"] = line.split("Data Interval Type")[-1].rstrip("|").strip()

    elif line.startswith("Data Type"):
        header_data["data_type"] = line.split("Data Type")[-1].rstrip("|").strip()


def read_records(file, read_all=False):
    records = []
    prev_timestamp = None
    section_finish = False
    header_data = {
        "format": None,
        "source": None,
        "station": None,
        "iaga_code": None,
        "interval": None,
        "data_type": None
    }

    for line in file:
        stripped = line.strip()
        data = parse_dst_line(stripped)

        # Drop invalid data
        if data == "invalid":
            continue

        # ----------------------
        # HEADER PARSING
        # ----------------------
    
        if data == "header":
            if not read_all and section_finish:
                yield {
                    "header": header_data,
                    "data": np.array(records, dtype=object)
                }
                records = []
                prev_timestamp = None
                section_finish = False
                header_data = {
                    "format": None,
                    "source": None,
                    "station": None,
                    "iaga_code": None,
                    "interval": None,
                    "data_type": None
                }
            hader_parser(header_data, stripped)

        # ----------------------
        # DATA PARSING
        # ----------------------

        else:
            current_timestamp = data["timestamp"]
            section_finish = True

            # ----------------------
            # FILL MISSING DATA
            # ----------------------
            if prev_timestamp is not None:
                expected_timestamp = prev_timestamp + timedelta(hours=1)

                while current_timestamp > expected_timestamp:
                    records.append({
                        "timestamp": expected_timestamp,
                        "dst_nT": None
                    })
                    expected_timestamp += timedelta(hours=1)

            records.append(data)

            prev_timestamp = current_timestamp

    yield {
        "header": header_data,
        "data": np.array(records, dtype=object)
    }
