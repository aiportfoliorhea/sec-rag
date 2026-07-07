# clients.py
from dotenv import load_dotenv
from anthropic import Anthropic
import cohere
import os
load_dotenv()
anthropic_client = Anthropic()
cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])