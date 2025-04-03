from openai import AzureOpenAI
import os

client = AzureOpenAI(
    azure_endpoint=os.getenv("ENDPOINT_URL", "YOUR_AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY"),
    api_version="YOUR_AZURE_API_VERSION",
)

class AIRecommendationAgent:
    """Decides inventory actions based on forecasted sales."""
    def decide_action(self, forecast_data, category):
        chat_prompt = [
            {"role": "system", "content": "You are an AI inventory assistant."},
            {"role": "user", "content": f"Forecasted Sales: {forecast_data}, Material Type: {category}, Recommend action."}
        ]

        completion = client.chat.completions.create(model=os.getenv("DEPLOYMENT_NAME"), messages=chat_prompt, max_tokens=10)
        return completion.choices[0].message.content.strip()
