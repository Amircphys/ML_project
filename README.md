# HeartFailureML

Installation:

```
python -m venv env
source env/bin/activate
pip install -e .
python src/train_pipeline.py configs/train_config.yml
```

Docker
```
python setup.py sdist
docker build -t  heart_failure:v1 .
docker run -e CLEARML_API_HOST=${CLEARML_API_HOST} -e CLEARML_API_ACCESS_KEY=${CLEARML_API_ACCESS_KEY} -e CLEARML_API_SECRET_KEY=${CLEARML_API_SECRET_KEY} -e LOGGER_LEVEL=${LOGGER_LEVEL} --name clearml_v1 heart_failure:v1
```


