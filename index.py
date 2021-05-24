from app import app, server
from callbacks import router, callback_navigation, callbacks_core

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False, port=8050)
