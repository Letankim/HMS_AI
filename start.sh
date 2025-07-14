#!/bin/bash
uvicorn nutrition_prediction_apiV1:app --host 0.0.0.0 --port $PORT
