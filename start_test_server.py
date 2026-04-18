import http.server
import socketserver
import webbrowser
import os

def start_local_server():
    """Khởi động local web server để test HTML"""
    
    # Tìm port trống
    port = 8080
    while True:
        try:
            httpd = socketserver.TCPServer(("", port), http.server.SimpleHTTPRequestHandler)
            print(f"🌐 Local server started on http://localhost:{port}")
            print(f"📁 Open browser: http://localhost:{port}/test_api_browser.html")
            
            # Mở browser tự động
            webbrowser.open(f'http://localhost:{port}/test_api_browser.html')
            
            # Chạy server
            httpd.serve_forever()
        except OSError:
            port += 1  # Thử port khác nếu bị占用

if __name__ == "__main__":
    start_local_server()
