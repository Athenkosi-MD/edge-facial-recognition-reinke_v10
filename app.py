import logging
import os

# Set up logging to see all import errors
logging.basicConfig(level=logging.DEBUG)

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(port=5000, debug=True)
