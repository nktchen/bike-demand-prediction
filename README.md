# Bike Demand Prediction Project

Проект по курсу ML в Центральном университете.

End-to-end ML-проект для прогнозирования почасового спроса на велосипеды. Модель может использоваться как MVP для планирования спроса и перераспределения велосипедного парка.

- Обучена регрессионная модель для прогнозирования спроса на велосипеды.
- Проведен полный цикл работы с данными: EDA, отбор признаков, генерация признаков, выбор модели через CV и оценка качества.
- Есть работа с признаками: создание временных, погодных, календарных и лаговых признаков и их отбор.
- Выполнена интерпретация итоговой модели с помощью библиотеки SHAP.
- Качество модели улучшено с baseline `R2 = 0.56`, `RMSE = 145.8` до `R2 = 0.928`, `RMSE = 59.02` на валидации.
- Написан сервис на FastAPI для инференса модели.
- Сервис контейнеризован с помощью Docker.

## Live demo

Можно протестировать модель в браузере! 

Перейдите по ссылке: [https://bike-ml-pred.nktchen.tech/predict_now](https://bike-ml-pred.nktchen.tech/predict_now).
Данные для модели берутся из [open-meteo.com](open-meteo.com).

Также можно отправить свой POST запрос на [https://bike-ml-pred.nktchen.tech/predict](https://bike-ml-pred.nktchen.tech/predict). Ожидаемый формат входных данных:
- timestamp: datetime
- temp_c: float #not normalisied
- feels_like_c: float #not normalisied
- humidity_percent: float = Field(ge=0, le=100)
- wind_speed_kmh: float = Field(ge=0)
- weathersit: int = Field(ge=1, le=3)
- holiday: int = Field(ge=0, le=1)

Пример тела запроса: 
```json
{
  "timestamp": "2012-12-29T17:00:00",
  "temp_c": 18.14,
  "feels_like_c": 28.12,
  "humidity_percent": 79.0,
  "wind_speed_kmh": 7.0015,
  "weathersit": 1,
  "holiday": 0
}
```

curl-запрос:
```bash
curl -X POST "https://bike-ml-pred.nktchen.tech/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2012-12-29T17:00:00",
    "temp_c": 18.14,
    "feels_like_c": 28.12,
    "humidity_percent": 79.0,
    "wind_speed_kmh": 7.0015,
    "weathersit": 1,
    "holiday": 0
  }'
```


## Модель

В качестве итоговой модели используется `CatBoostRegressor`.

Основные группы признаков:

- календарные признаки: месяц, час, день недели, праздник, рабочий день;
- погодные признаки: температура, ощущаемая температура, влажность, скорость ветра, тип погоды;
- циклические признаки времени: sin/cos для часа, дня недели и месяца;
- контекстные признаки: час пик, ночь, холодный сезон, хорошая/плохая погода;
- исторические лаги: спрос 24 часа назад и 168 часов назад.

## Структура проекта

```text
src/train.py           обучение модели
src/make_demo_data.py  подготовка demo history и demo requests
src/features.py        генерация признаков
src/predict.py         загрузка модели и демо инференс без поднятия приложения
src/app.py             FastAPI-приложение
models/                сохраненная модель
data/demo/             данные для demo-инференса
Dockerfile             контейнеризация API
.sh                    скрипты для удобного развертывания
```

## Быстрый запуск

Создайте окружение и установите зависимости: 
```bash
./setup_env.sh
```
Подготовьте данные и обучите модель:
```bash
./prepare_data_and_train.sh
```
Запустите API:

```bash
.venv/bin/uvicorn src.app:app
```
API будет доступно по адресу:
```text
http://127.0.0.1:8000
```

Данные для запроса можно найти в data/demo/demo_requests.json