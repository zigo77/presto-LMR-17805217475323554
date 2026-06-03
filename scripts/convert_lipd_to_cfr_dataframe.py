#!/usr/bin/env python3
"""
Convert LiPD pickle format to CFR-compatible pandas DataFrame pickle.

This script extracts proxy data from LiPD (Linked Paleo Data) format and converts
it to a pandas DataFrame that CFR can load directly via its fetch() method.

Usage:
    python convert_lipd_to_cfr_dataframe.py lipd.pkl lipd_cfr.pkl
"""

import sys
import pickle
import pandas as pd
import numpy as np
from collections import OrderedDict


def extract_proxy_data(proxy_dict, proxy_id):
    """
    Extract proxy data from a single LiPD entry.

    Args:
        proxy_dict: Dictionary containing LiPD proxy metadata and data
        proxy_id: String identifier for the proxy

    Returns:
        Dictionary with CFR-compatible proxy data, or None if extraction fails
    """
    try:
        # Extract geographic metadata
        geo = proxy_dict.get('geo', {})
        if isinstance(geo, dict):
            geometry = geo.get('geometry', {})
            if isinstance(geometry, dict):
                coords = geometry.get('coordinates', [None, None])
                lon = coords[0]
                lat = coords[1]
            else:
                # Try alternative structure
                lat = geo.get('latitude', geo.get('meanLat', None))
                lon = geo.get('longitude', geo.get('meanLon', None))
        else:
            lat = None
            lon = None

        # Normalize longitude to 0-360 range (CFR standard)
        if lon is not None and lon < 0:
            lon = lon + 360

        # Extract archive type
        archive_type = proxy_dict.get('archiveType', 'unknown')
        if isinstance(archive_type, str):
            archive_type = archive_type.lower()
        else:
            archive_type = 'unknown'

        # Navigate to paleoclimate data
        paleo_data = proxy_dict.get('paleoData', {})
        if not isinstance(paleo_data, (dict, OrderedDict)):
            return None

        # Get first paleo dataset
        paleo0 = paleo_data.get('paleo0', {})
        if not isinstance(paleo0, (dict, OrderedDict)):
            # Try getting first key if paleo0 doesn't exist
            if len(paleo_data) > 0:
                first_key = list(paleo_data.keys())[0]
                paleo0 = paleo_data[first_key]
            else:
                return None

        # Get measurement table
        measurement_table = paleo0.get('measurementTable', {})
        if not isinstance(measurement_table, (dict, OrderedDict)):
            return None

        # Get first measurement table entry
        if len(measurement_table) == 0:
            return None

        first_table_key = list(measurement_table.keys())[0]
        table = measurement_table[first_table_key]

        if not isinstance(table, dict):
            return None

        columns = table.get('columns', {})
        if not isinstance(columns, (dict, OrderedDict)):
            return None

        # Extract time and value data
        time_data = None
        value_data = None
        proxy_type = None
        value_name = None

        for col_name, col_dict in columns.items():
            if not isinstance(col_dict, dict):
                continue

            var_name = col_dict.get('variableName', '').lower()
            values = col_dict.get('values', [])

            # Identify time column
            if var_name in ['year', 'age', 'time', 'yr']:
                time_data = values
            # Identify value column (various proxy measurements)
            elif var_name in ['d18o', 'd18O', 'srca', 'SrCa', 'trw', 'TRW',
                             'mxd', 'MXD', 'dd', 'dD', 'temperature', 'temp',
                             'accumulation', 'thickness', 'mgca', 'MgCa',
                             'uk37', 'UK37', 'tex86', 'TEX86']:
                value_data = values
                proxy_type = var_name
                value_name = col_dict.get('longName', var_name)

        # Validate extracted data
        if time_data is None or value_data is None:
            return None

        if len(time_data) == 0 or len(value_data) == 0:
            return None

        if len(time_data) != len(value_data):
            # Truncate to minimum length
            min_len = min(len(time_data), len(value_data))
            time_data = time_data[:min_len]
            value_data = value_data[:min_len]

        # Construct ptype (archive.proxy format)
        if proxy_type:
            # Standardize common proxy type names
            proxy_type_map = {
                'd18o': 'd18O',
                'srca': 'SrCa',
                'trw': 'TRW',
                'mxd': 'MXD',
                'dd': 'dD',
                'mgca': 'MgCa',
                'uk37': 'UK37',
                'tex86': 'TEX86'
            }
            proxy_type_std = proxy_type_map.get(proxy_type.lower(), proxy_type)
            ptype = f"{archive_type}.{proxy_type_std}"
        else:
            ptype = f"{archive_type}.unknown"

        # Construct CFR-compatible record with exact PAGES2k column names
        return {
            'paleoData_pages2kID': proxy_id,
            'dataSetName': proxy_id,  # Dataset name
            'archiveType': archive_type,
            'geo_meanLat': float(lat) if lat is not None else np.nan,
            'geo_meanLon': float(lon) if lon is not None else np.nan,
            'geo_meanElev': 0.0,  # Default; LiPD may not always have elevation
            'year': time_data,  # CFR expects 'year' not 'time'
            'paleoData_values': value_data,  # CFR expects 'paleoData_values' not 'value'
            'paleoData_variableName': proxy_type if proxy_type else 'unknown',
            'paleoData_units': 'permil' if 'd18O' in str(proxy_type).lower() or 'dD' in str(proxy_type) else 'unknown',
            'paleoData_proxy': proxy_type if proxy_type else 'unknown',
            'paleoData_ProxyObsType': ptype,  # Combined archive.proxy format
        }

    except Exception as e:
        print(f"  Error extracting {proxy_id}: {e}")
        return None


def convert_lipd_to_dataframe(lipd_pkl_path):
    """
    Convert LiPD pickle file to pandas DataFrame.

    Args:
        lipd_pkl_path: Path to LiPD pickle file

    Returns:
        pandas DataFrame with CFR-compatible structure
    """
    print(f"Loading LiPD pickle: {lipd_pkl_path}")

    with open(lipd_pkl_path, 'rb') as f:
        data = pickle.load(f)

    # LiPD structure: {'D': {proxy_id: proxy_data, ...}}
    proxy_dict = data.get('D', {})

    if not proxy_dict:
        raise ValueError("No 'D' key found in pickle file. Is this a valid LiPD file?")

    print(f"Found {len(proxy_dict)} proxies in LiPD file")
    print()

    records = []
    skipped = 0

    for proxy_id, proxy_data in proxy_dict.items():
        record = extract_proxy_data(proxy_data, proxy_id)

        if record:
            records.append(record)
            print(f"  [OK] {proxy_id}: {record['paleoData_ProxyObsType']}, {len(record['year'])} points")
        else:
            skipped += 1
            print(f"  [SKIP] {proxy_id}: missing data or extraction failed")

    print()
    print(f"Successfully extracted: {len(records)} proxies")
    print(f"Skipped: {skipped} proxies")

    if len(records) == 0:
        raise ValueError("No valid proxy records extracted!")

    # Create DataFrame
    df = pd.DataFrame(records)

    print()
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    print("Proxy type distribution:")
    print(df['paleoData_ProxyObsType'].value_counts())
    print()
    print("Sample of first 3 proxies:")
    for idx in range(min(3, len(df))):
        row = df.iloc[idx]
        print(f"  {row['paleoData_pages2kID']}:")
        print(f"    proxy type: {row['paleoData_ProxyObsType']}")
        print(f"    lat/lon: {row['geo_meanLat']:.2f}, {row['geo_meanLon']:.2f}")
        print(f"    time points: {len(row['year'])}")
        print(f"    time range: {min(row['year']):.1f} - {max(row['year']):.1f}")
        print()

    return df


def main():
    """Main entry point for the converter."""

    if len(sys.argv) != 3:
        print(__doc__)
        print("\nExample:")
        print("  python convert_lipd_to_cfr_dataframe.py lipd.pkl lipd_cfr.pkl")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        # Convert LiPD to DataFrame
        df = convert_lipd_to_dataframe(input_path)

        # Save as pickle with protocol 4 for better compatibility
        print("="*60)
        print(f"Saving DataFrame pickle: {output_path}")
        print(f"Using pickle protocol 4 for pandas version compatibility...")
        df.to_pickle(output_path, protocol=4)

        # Verify the saved file
        print(f"Verifying saved file...")
        df_test = pd.read_pickle(output_path)

        print()
        print("[OK] Successfully saved and verified!")
        print(f"[OK] Output file: {output_path}")

        import os
        file_size_kb = os.path.getsize(output_path) / 1024
        print(f"[OK] File size: {file_size_kb:.1f} KB")
        print()
        print("This file is ready to use with CFR!")
        print()
        print("Next steps:")
        print(f"  1. Update lmr_configs.yml: proxydb_path: /app/{output_path.split('/')[-1]}")
        print(f"  2. Test loading: python -c \"import cfr; pdb = cfr.ProxyDatabase().fetch('{output_path}'); print(pdb.nrec)\"")
        print(f"  3. Upload to your server or GitHub for workflow access")

    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
