FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from src.models.forecaster import train_demand_forecaster; train_demand_forecaster()"
ENTRYPOINT ["python", "src/cli/forecast_cli.py"]
