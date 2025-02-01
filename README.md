# Aircraft Maintenance

The random.csv is what you put into the app for prediction.

To run, there are 2 ways:

### Using Docker

1. `docker build -t pred-maintenance .`
2. `docker run -p 8501:8501 pred-maintenance`

### The regular way
1. `streamlit run main.py`