Построение и запуск контейнера:

```
docker build -t online_inference:v1 .
docker run -d -p 8000:8000 online_inference:v1
```

Тестировние:
```
pytest test_main.py
```

Пример работы модели:
```
python make_request.py
```