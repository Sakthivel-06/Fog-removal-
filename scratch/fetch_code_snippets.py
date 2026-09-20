import urllib.request

def fetch_file(repo, branch, path, max_chars=3000):
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path}"
    req = urllib.request.Request(url, headers={'User-Agent': 'ResearchBot'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            content = r.read().decode('utf-8', errors='ignore')
            print(f"=== {repo}/{path} ({len(content)} chars) ===")
            print(content[:max_chars])
            print("\n-----------------------------------\n")
    except Exception as e:
        print(f"Error fetching {repo}/{path}:", e)

if __name__ == '__main__':
    fetch_file("fanjunkai1/DCL", "main", "networks/beta_decoder.py", 2000)
    fetch_file("fanjunkai1/DCL", "main", "networks/dehaze.py", 2500)
    fetch_file("fanjunkai1/DCL", "main", "networks/losses.py", 2500)
    fetch_file("fanjunkai1/DVD", "main", "archs/DVD_arch.py", 2500)
