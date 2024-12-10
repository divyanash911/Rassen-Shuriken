import requests
from transformers import pipeline

def get_labels(infos):
    
    """Replace with actual business logic

    Returns:
        score: % of correct matches.
        incorrect: number of incorrect matches.
    """
    messages = [
    {"role": "user", "content": "Who are you?"},
    ]
    classifier = pipeline("text-generation", model="aiplanet/buddhi-indic")  

    score = 0
    incorrect = []
    labels = {}

    for info in infos:
        info_query = {"role":"user","content":f"You are a content moderator for github issues. Your task is to label issues as inappropriate or not. Given an issue return only NSFW if appropriate else return SFW. Nothing else.\nIssue\n{info}"}
        if classifier(info_query)[0]['label'] == 'NSFW':
            labels[info] = 'Inappropriate'
            score+=1
        else:
            incorrect.append(info)

    return score,incorrect

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
    
score,incorrect = get_labels(infos)
        
print(f"Total predicted inappropriate : {(score/len(infos))*100}")
print(f"Incorrect predictions:{incorrect}")