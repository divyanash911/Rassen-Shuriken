KEY = 'AIzaSyAstDCNB7JplgZL0M7aFDxGLf8HGEs3MNU'

TEMPLATE = 'You are a content moderator for github issues. Your task is to label issues as inappropriate or not. Inappropriate issue could be anything which is demeaning, harmful or irrelevant for a github issue. It can be in non-english languages. Given an issue return only NSFW if appropriate else return SFW. Nothing else.'

import os
import time 
import requests
import google.generativeai as genai

# Configuration
TOKEN = "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"
OWNER = "virtual-labs"
REPO = "bugs-virtual-labs"
LABEL = "Inappropriate"


# API URL
url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
params = {
    "labels": LABEL,   # Specify the label
    "state": "closed",   # Filter by state ('open', 'closed', or 'all')
    "per_page": 100,   # Number of issues per page (max: 100)
    "page": 1          # Pagination support
}


# Authorization Header
headers = {
    "Authorization": f"token {TOKEN}"
}

# Make the Request
infos = []
response = requests.get(url, headers=headers, params=params)
if response.status_code == 200:
    issues = response.json()
    for issue in issues:
        content = issue['body'].split("\n")
        for line in content:
            if 'Additional info' in line:
                infos.append(line.replace('Additional info-',''))
else:
    print(f"Error: {response.status_code} - {response.text}")

genai.configure(api_key=KEY)

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
)

chat_session = model.start_chat(
  history=[
  ]
)
for info in infos:
    
	response = chat_session.send_message(f"{TEMPLATE}\nIssue\n{info}")
	print(f"Issue - {info}\nGenerated verdict - {response.text}")
	time.sleep(2)