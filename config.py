import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class."""
    # API Keys
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

    # Application Settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # Add more configuration variables as needed
    
    @classmethod
    def get_config(cls):
        """Get the configuration instance."""
        return cls() 