from database.sql_query_agent import azure_sql_gm3
from agents.classification import AIDrivenClassificationAgent
from agents.forecast import MovingAverageForecastAgent, SlowMovingForecastAgent, SeasonalForecastAgent
from agents.recommendation import AIRecommendationAgent

def run_inventory_pipeline(material_code):
    """Runs inventory forecasting pipeline using SQL data."""
    print(f"\n🔍 Fetching sales data for Material: {material_code}...")

    query_result = azure_sql_gm3(f"Get details of Material '{material_code}' from Consumption.")

    if not query_result:
        print(f"❌ No data found for {material_code}")
        return

    sales_data = [entry["sales"] for entry in query_result]

    classification_agent = AIDrivenClassificationAgent()
    recommendation_agent = AIRecommendationAgent()

    category = classification_agent.classify_material(sales_data)

    # Select forecast model based on classification
    if "fast" in category.lower():
        forecast = MovingAverageForecastAgent().forecast(sales_data)
    elif "slow" in category.lower():
        forecast = SlowMovingForecastAgent().forecast(sales_data)
    else:
        forecast = SeasonalForecastAgent().forecast(sales_data)

    decision = recommendation_agent.decide_action(forecast, category)

    # Print the result
    print(f"\n📌 Material: {material_code}")
    print(f"📊 Category: {category}")
    print(f"📈 Forecast (Next 6 Months): {forecast}")
    print(f"📦 Recommended Action: {decision}")

# Run the pipeline for a test material
if __name__ == "__main__":
    run_inventory_pipeline("GM21882483")
