import urllib.request
import urllib.parse
import re
from html import unescape

def ddg_search(q):
    url = 'https://html.duckduckgo.com/html/?q=' + urllib.parse.quote(q)
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            html = r.read().decode('utf-8', errors='ignore')
            # Extract results
            results = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
            links = re.findall(r'<a class="result__url[^"]*"[^>]*href="([^"]+)"', html, re.DOTALL)
            titles = re.findall(r'<h2 class="result__title">.*?<a[^>]*>(.*?)</a>', html, re.DOTALL)
            print(f"=== DDG: {q} ===")
            for i in range(min(len(titles), 4)):
                t = unescape(re.sub(r'<[^>]+>', '', titles[i])).strip()
                s = unescape(re.sub(r'<[^>]+>', '', results[i])).strip() if i < len(results) else ""
                u = links[i] if i < len(links) else ""
                print(f"Title: {t}")
                print(f"URL: {u}")
                print(f"Snippet: {s[:250]}\n")
    except Exception as e:
        print('Error:', e)

if __name__ == '__main__':
    ddg_search('"Driving-Video Dehazing" CVPR 2024')
    ddg_search('DCL "AAAI 2025" dehazing')
    ddg_search('"Driving-Video Dehazing" github')
    ddg_search('DCL dehazing github')
