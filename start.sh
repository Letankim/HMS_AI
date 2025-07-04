#!/bin/bash
uvicorn nutrition_prediction_api.v1:app --host 0.0.0.0 --port 10000

