import urllib.request
import json

def list_repo_tree(repo):
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode())
            print(f"=== File Tree for {repo} ===")
            py_files = [x['path'] for x in data.get('tree', []) if x['path'].endswith('.py')]
            for f in py_files[:25]:
                print(" ", f)
    except Exception as e:
        print(f"Error listing tree for {repo}:", e)

if __name__ == '__main__':
    list_repo_tree("fanjunkai1/DVD")
    list_repo_tree("fanjunkai1/DCL")
