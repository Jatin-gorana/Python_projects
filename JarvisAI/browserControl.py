import subprocess
import threading
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# Dictionary to track browser processes
browser_processes = {}

# Supported browsers and their commands
BROWSER_COMMANDS = {
    'chrome': lambda url: ['chrome', url] if os.name != 'nt' else ['start', 'chrome', url],
    'firefox': lambda url: ['firefox', url] if os.name != 'nt' else ['start', 'firefox', url],
}

def start_browser(browser, url):
    if browser in BROWSER_COMMANDS:
        cmd = BROWSER_COMMANDS[browser](url)
        # Use shell=True for Windows, False otherwise
        proc = subprocess.Popen(cmd, shell=(os.name == 'nt'))
        browser_processes[browser] = proc
        return True
    return False

def stop_browser(browser):
    proc = browser_processes.get(browser)
    if proc:
        proc.terminate()
        proc.wait()
        browser_processes.pop(browser)
        return True
    return False

@app.route('/start')
def api_start_browser():
    browser = request.args.get('browser')
    url = request.args.get('url')
    if start_browser(browser, url):
        return jsonify({'status': 'started', 'browser': browser, 'url': url})
    else:
        return jsonify({'status': 'error', 'message': 'Unsupported browser'}), 400

@app.route('/stop')
def api_stop_browser():
    browser = request.args.get('browser')
    if stop_browser(browser):
        return jsonify({'status': 'stopped', 'browser': browser})
    else:
        return jsonify({'status': 'error', 'message': 'Browser not running'}), 400

@app.route('/geturl')
def api_get_url():
    browser = request.args.get('browser')
    # Only demonstration: Not trivial to fetch URL of running browser without extensions or OS hooks
    # You'd need browser extensions or advanced OS integration for actual current tab retrieval
    return jsonify({'status': 'error', 'message': 'Fetching current URL not implemented'}), 501

@app.route('/cleanup')
def api_cleanup():
    browser = request.args.get('browser')
    # Placeholder: Real cleanup would clear browser cache/history, needs OS-specific handling
    return jsonify({'status': 'success', 'browser': browser, 'message': 'Cleanup triggered (mock)'})

if __name__ == '__main__':
    app.run(port=5000)
