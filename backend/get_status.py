import urllib.request, json
try:
    resp = urllib.request.urlopen("http://127.0.0.1:5066/api/pipeline-status")
    print(resp.read().decode('utf-8'))
except Exception as e:
    print(str(e))
