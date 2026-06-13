FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV AGENT_ROUTER_CONFIG=/app/config.yaml

EXPOSE 8000

CMD ["python", "-m", "agent_router.server"]
