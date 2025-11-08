from flask import Flask, send_from_directory

app = Flask(__name__, static_folder="frontend")  # frontend is where index.html is

# Serve the dashboard
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------- Detect Endpoint ----------------
@app.route("/detect")
def detect():
    # Your detection code here
    # Return JSON like:
    return jsonify({
        "face": "John Doe",
        "objects": ["gun"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
