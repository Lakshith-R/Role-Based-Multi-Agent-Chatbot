import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from collections import defaultdict
import streamlit as st

# ---------------- SETUP ----------------

# Define the scope for accessing Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Authenticate using credentials from Streamlit secrets
creds_dict = {
    "type": st.secrets["gcp_service_account"]["type"],
    "project_id": st.secrets["gcp_service_account"]["project_id"],
    "private_key_id": st.secrets["gcp_service_account"]["private_key_id"],
    "private_key": st.secrets["gcp_service_account"]["private_key"].replace("\\n", "\n"),
    "client_email": st.secrets["gcp_service_account"]["client_email"],
    "client_id": st.secrets["gcp_service_account"]["client_id"],
    "auth_uri": st.secrets["gcp_service_account"]["auth_uri"],
    "token_uri": st.secrets["gcp_service_account"]["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["gcp_service_account"]["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["gcp_service_account"]["client_x509_cert_url"]
}

creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
client = gspread.authorize(creds)

# Access your sheet by name
SHEET_NAME = "WorkflowLogs"
sheet = client.open(SHEET_NAME).sheet1

# Get all rows as a list of dictionaries
data = sheet.get_all_records()
df = pd.DataFrame(data)

# ---------------- ANALYSIS ----------------

total_queries = len(df)
fallbacks = df[df["Fallback Used"] == "Yes"]
fallback_count = len(fallbacks)
fallback_rate = (fallback_count / total_queries * 100) if total_queries else 0

avg_latency_by_agent = df.groupby("Agent")["Latency"].mean()
query_counts = df["Agent"].value_counts()

# Optional: Routing accuracy if you added "Expected Agent" column manually
if "Expected Agent" in df.columns:
    routing_matches = df[df["Agent"] == df["Expected Agent"]]
    routing_accuracy = (len(routing_matches) / len(df)) * 100
else:
    routing_accuracy = None

# ---------------- OUTPUT ----------------

print("📊 Chatbot Evaluation Summary\n")
print(f"✅ Total Queries Logged: {total_queries}")
print(f"🛡️ Fallbacks Triggered: {fallback_count} ({fallback_rate:.2f}%)\n")

print("📌 Query Count per Agent:")
for agent, count in query_counts.items():
    print(f"   - {agent}: {count} queries")

print("\n📌 Average Latency per Agent:")
for agent, latency in avg_latency_by_agent.items():
    print(f"   - {agent}: {latency:.2f} seconds")

if routing_accuracy is not None:
    print(f"\n🎯 Routing Accuracy: {routing_accuracy:.2f}%")
else:
    print("\n🎯 Routing Accuracy: Not available (no 'Expected Agent' column)")
