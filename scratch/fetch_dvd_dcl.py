import urllib.request
import json

def fetch_gh_repo_info(repo):
    base_url = f"https://api.github.com/repos/{repo}"
    req = urllib.request.Request(base_url, headers={'User-Agent': 'ResearchBot'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            print(f"=== REPO: {repo} ===")
            print("Description:", data.get("description"))
            print("License:", data.get("license"))
            print("Default branch:", data.get("default_branch"))
    except Exception as e:
        print(f"Error fetching {repo}:", e)

def fetch_gh_file(repo, branch, path):
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            content = r.read().decode('utf-8', errors='ignore')
            print(f"--- File: {repo}/{path} ({len(content)} chars) ---")
            print(content[:1500])
            print("\n...[truncated]...\n")
            return content
    except Exception as e:
        print(f"Error fetching {repo}/{path}:", e)
        return ""

if __name__ == '__main__':
    fetch_gh_repo_info("fanjunkai1/DVD")
    fetch_gh_file("fanjunkai1/DVD", "main", "README.md")
    fetch_gh_file("fanjunkai1/DVD", "main", "requirements.txt")

    fetch_gh_repo_info("fanjunkai1/DCL")
    fetch_gh_file("fanjunkai1/DCL", "main", "README.md")
    fetch_gh_file("fanjunkai1/DCL", "main", "requirements.txt")
