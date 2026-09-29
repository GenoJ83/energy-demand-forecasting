"""CLI tool to display upcoming 24-hour smart grid load estimates."""
import argparse
from src.data.generator import generate_hourly_load_data
from src.models.forecaster import train_demand_forecaster
from src.inference.predict_future import DemandPredictor

def main():
    parser = argparse.ArgumentParser(description="Smart Grid Energy Forecaster")
    parser.add_argument("--horizon", type=int, default=6, help="Forecast horizon in hours")
    args = parser.parse_args()

    train_demand_forecaster()
    predictor = DemandPredictor()
    df = generate_hourly_load_data(10)
    pred_mw = predictor.forecast_next_step(df)

    print(f"Forecast for T+1 Hour: {pred_mw} Megawatts (Horizon: {args.horizon}h)")

if __name__ == "__main__":
    main()
