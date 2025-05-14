from app import app
import platform

if __name__ == "__main__":
    try:
        # Simplified approach for Windows
        if platform.system() == "Windows":
            # Use minimal configuration that avoids socket issues
            app.run(debug=False)
        else:
            app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        print(f"Error starting the application: {e}")
        print("Trying alternative method...")
        # Fallback method
        from werkzeug.serving import run_simple
        run_simple('127.0.0.1', 5000, app, use_debugger=False, use_reloader=False)
