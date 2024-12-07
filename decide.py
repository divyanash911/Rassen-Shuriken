import os

def blackbox_function(title, body, author):
    """
    Blackbox function to decide if an issue is inappropriate.
    Replace this with the actual logic or model.
    """
    
    return False

def main():
    issue_title = os.getenv('ISSUE_TITLE', '')
    issue_body = os.getenv('ISSUE_BODY', '')
    issue_author = os.getenv('ISSUE_AUTHOR', '')

    is_inappropriate = blackbox_function(issue_title, issue_body, issue_author)
    print("true" if is_inappropriate else "false")

if __name__ == "__main__":
    main()
