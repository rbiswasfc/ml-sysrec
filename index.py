from app import app, server
from callbacks import router, callback_navigation

if __name__ == "__main__":
    app.run_server(debug=True)
