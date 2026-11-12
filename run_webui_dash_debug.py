#!/usr/bin/env python
"""
run_webui_dash_debug.py - Debug version with timeout and better error handling
"""

import argparse
import sys
import os
import socket
import signal
import threading
import time

from webui.app_dash import run_app  


class TimeoutHandler:
    """Handle analysis timeouts to prevent hanging"""
    
    def __init__(self, timeout_seconds=300):  # 5 minutes default
        self.timeout_seconds = timeout_seconds
        self.timer = None
        
    def start_timeout(self):
        """Start the timeout timer"""
        if self.timer:
            self.timer.cancel()
        
        self.timer = threading.Timer(self.timeout_seconds, self.timeout_handler)
        self.timer.start()
        print(f"⏰ Analysis timeout set to {self.timeout_seconds} seconds")
        
    def cancel_timeout(self):
        """Cancel the timeout timer"""
        if self.timer:
            self.timer.cancel()
            print("✅ Analysis completed before timeout")
            
    def timeout_handler(self):
        """Handle timeout - print warning and continue"""
        print(f"\n⚠️ WARNING: Analysis has been running for {self.timeout_seconds} seconds")
        print("This might indicate a hang in one of the analysts.")
        print("Common causes:")
        print("  - Social Analyst stuck on Reddit API calls")
        print("  - Network timeout in data fetching")
        print("  - Infinite loop in analyst logic")
        print("\nThe web UI will continue running. You can:")
        print("  1. Wait longer if this is expected")
        print("  2. Stop the analysis using the web UI")
        print("  3. Restart the application")


def find_available_port(start_port, end_port=None):
    """Find an available port in the given range"""
    if end_port is None:
        end_port = start_port + 100
    
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return port
            except OSError:
                continue
    return None


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TradingAgents Dash Web UI (Debug Mode)")
    
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--share", action="store_true", help="Share the app publicly")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument("--max-threads", type=int, default=40, help="Maximum number of threads")
    parser.add_argument("--timeout", type=int, default=300, help="Analysis timeout in seconds")
    
    return parser.parse_args()


def main():
    """Run the Dash web UI with debug features"""
    args = parse_args()
    
    # Find an available port
    port = find_available_port(args.port)
    if port is None:
        print(f"❌ Error: Could not find an available port between {args.port} and {args.port + 100}")
        return 1
    
    if port != args.port:
        print(f"⚠️ Port {args.port} is in use. Using port {port} instead.")

    # Setup timeout handler
    timeout_handler = TimeoutHandler(args.timeout)
    
    # Run the app
    try:
        result = run_app(
            port=port,
            share=args.share,
            server_name=args.server_name,
            debug=args.debug,
            max_threads=args.max_threads
        )
        return result
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
        return 0
    except Exception as e:
        print(f"\n❌ Error running app: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 