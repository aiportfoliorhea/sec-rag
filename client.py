# clients.py
from anthropic import Anthropic
import cohere
import os

anthropic_client = Anthropic()
cohere_client = cohere.Client(api_key=os.environ["COHERE_API_KEY"])