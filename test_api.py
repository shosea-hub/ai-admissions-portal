"""
test_api.py — Simple test client for the AI Admission System API.
Sends sample applicants from sample_applicants.csv to /process_application
and prints the AI model response for each.
"""

import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:5000"

def main():
    csv_path = "sample_applicants.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Could not find {csv_path}")
        return

    print(f"📤 Sending {len(df)} applicants to {BASE_URL}/process_application\n")

    for i, row in df.iterrows():
        data = {
            "fullname": row.get("fullname", ""),
            "program": row.get("program", ""),
            "jamb": row.get("jamb", ""),
            "cgpa": row.get("cgpa", ""),
            "dob": row.get("dob", ""),
        }
        try:
            resp = requests.post(f"{BASE_URL}/process_application", data=data)
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ {data['fullname']}: {result['status']} "
                      f"({result['eligibilityScore']}% confidence)")
            else:
                print(f"⚠️ {data['fullname']}: Server responded with {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"❌ Error sending {data['fullname']}: {e}")

    print("\n✅ Test completed.")

if __name__ == "__main__":
    main()
