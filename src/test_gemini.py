import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

interaction = client.interactions.create(
    model="gemini-3.7-flash",
    input="In one sentence, what is the HIPAA Security Rule?",
)

print(interaction.output_text)