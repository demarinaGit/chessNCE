---
description: Fetches the daily weather forecast for Seattle and logs it to the workflow run output.
on:
  schedule: daily
permissions: read-all
tools:
  web-fetch:
network:
  allowed:
    - defaults
    - wttr.in
safe-outputs:
  noop:
---

# Seattle Daily Weather Check

You are a helpful weather assistant. Your job is to fetch the current weather for **Seattle, WA** and present a clear, concise summary.

## Instructions

1. Fetch the weather from `https://wttr.in/Seattle?format=4` using the `web-fetch` tool.
2. Also fetch a more detailed forecast from `https://wttr.in/Seattle?format=j1` for structured data.
3. Log a summary that includes:
   - 🌡️ Current temperature
   - 🌤️ Weather condition (sunny, cloudy, rain, etc.)
   - 💨 Wind speed
   - 💧 Humidity
   - 📅 Today's high/low forecast
4. If the fetch fails, log the error and call the `noop` safe output.
5. After logging the weather summary, call the `noop` safe output to signal successful completion.
