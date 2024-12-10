import requests
from dotenv import load_dotenv

load_dotenv()

KEY = os.getenv('KEY')

TOKEN = os.getenv('TOKEN')
OWNER = "virtual-labs"
REPO = "bugs-virtual-labs"

import os
import google.generativeai as genai

genai.configure(api_key=KEY)



def predict_label(comment):
    
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

    response = chat_session.send_message("INSERT_INPUT_HERE")

    if 'NSFW' in response.text:
        return 'NSFW'
    
    return 'SFW'


def get_comments(issues):
    infos = []
    for issue in issues:
        curr_issue = {issue['number']: ''}
        flag = 0
        content = issue['body'].split("\n")
        for line in content:
            if 'Additional info' in line:
                flag = 1
            elif 'UserAgent' in line:
                flag = 0
                break

            if flag == 1 and 'UserAgent' not in line:
                curr_issue[issue['number']] += (line.replace('Additional info-', '').strip())
            elif flag == 1 and 'UserAgent' in line:
                break
        infos.append(curr_issue)

    print(infos)
    return infos


def get_labels(comments):

    labels = []
    for comment in comments:

        comment_str = list(comment.values())[0]
        label = predict_label(comment_str)
        labels.append({list(comment.keys())[0]: label})

    return labels

# API URL
url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues"
headers = {
    "Authorization": f"token {TOKEN}"
}

# get all the issues
def get_issues(url, headers):
    issues = []
    params = {
        "state": "open",
        "per_page": 100,
        "page": 1
    }
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        issues = response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
    return issues


issues = get_issues(url, headers)
labels = get_labels(get_comments(issues))

##if NSFW for any issue, label it as Inappropriate in github
for label in labels:
    issue_number = list(label.keys())[0]
    label = list(label.values())[0]
    if label == 'NSFW':
        url = f"https://api.github.com/repos/{OWNER}/{REPO}/issues/{issue_number}/labels"
        response = requests.post(url, headers=headers, json={"labels": ["Inappropriate"]})
        print(response.status_code)
        print(response.text)
        

