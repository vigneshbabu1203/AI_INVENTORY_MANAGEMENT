import os
from openai import AzureOpenAI

# Azure OpenAI Client Setup
endpoint = os.getenv("ENDPOINT_URL", "YOUR_AZURE_ENDPOINT")
deployment = os.getenv("DEPLOYMENT_NAME", "YOUR_AZURE_DEPLOYMENT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_API_KEY")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2024-08-01-preview",
)

class AIDrivenClassificationAgent:
    """Classifies materials based on sales data."""
    def classify_material(self, sales_data):
        chat_prompt = [
            {"role": "system", "content": "You are an AI specializing in inventory classification."},
            {"role": "user", "content": f"""
                Analyze the following sales data and classify the material:
                - Fast-Moving
                - Slow-Moving
                - Seasonal
                **Sales Data:** {list(sales_data)}
                Provide ONLY the category name.
            """}
        ]

        completion = client.chat.completions.create(
            model=deployment, messages=chat_prompt, max_tokens=5, temperature=0.3
        )
        return completion.choices[0].message.content.strip()
