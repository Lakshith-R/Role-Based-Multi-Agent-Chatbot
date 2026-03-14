import os
from pathlib import Path
import gspread
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Absolute paths — works regardless of what directory Streamlit is launched from
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent  # → Agentic_Student_Assistant-main/
OAUTH_CREDENTIALS_FILE = str(_PROJECT_ROOT / "logs" / "oauth_credentials.json")
TOKEN_FILE = str(_PROJECT_ROOT / "logs" / "token.json")

print(f"DEBUG: sheets_logger.py loading from: {__file__}")


def _get_oauth_creds() -> Credentials:
    """
    Load cached OAuth token or run browser-based login flow.
    The browser prompt only appears on the FIRST run; afterwards token.json is reused.
    """
    creds = None

    # Reuse saved token if it exists
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    # If no valid creds, run the OAuth flow (opens browser once)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print(f"DEBUG: OAUTH_CREDENTIALS_FILE='{OAUTH_CREDENTIALS_FILE}'")
            print(f"DEBUG: Exists: {os.path.exists(OAUTH_CREDENTIALS_FILE)}")
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for future runs
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return creds


def log_to_gsheet(
    timestamp, query, agent, curriculum_mode, latency, is_fallback: bool = False, result: str = ""
):  # pylint: disable=R0917
    """
    Log an interaction to Google Sheets using OAuth 2.0 (Desktop App flow).
    On the first call a browser window will open to authenticate with your Google account.
    Subsequent calls reuse the cached token in logs/token.json.
    """
    creds = _get_oauth_creds()
    client = gspread.authorize(creds)
    
    try:
        spreadsheet = client.open("WorkflowLogs")
        sheet = spreadsheet.sheet1
        
        # Add headers if the sheet is completely empty
        if not sheet.get_all_values():
            sheet.append_row([
                "Timestamp", "Query", "Agent", "Curriculum Mode", 
                "Latency", "Fallback Used", "Result"
            ])
            
    except gspread.exceptions.SpreadsheetNotFound:
        print("DEBUG: 'WorkflowLogs' sheet not found. Creating it now...")
        spreadsheet = client.create("WorkflowLogs")
        sheet = spreadsheet.sheet1
        # Add headers to the new sheet
        sheet.append_row([
            "Timestamp", "Query", "Agent", "Curriculum Mode", 
            "Latency", "Fallback Used", "Result"
        ])

    row = [
        timestamp,
        (query or "").replace("\n", " ").strip(),
        (agent or "").strip(),
        (curriculum_mode or "").strip(),
        round(latency, 2) if latency else "",
        "Yes" if is_fallback else "No",
        (result or "").replace("\n", " ").strip()[:500]
    ]
    sheet.append_row(row)
