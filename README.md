# MISIS-SUPER-RESOLUTION
Учебный проект для исследовательской работы по НИР. Включает в себя лаунчер моделей Real-ESRGAN и RT4KSR, а также калькулятор метрик.
## Требования
- Дистрибутив Linux или Windows с интеграцией WSL2 (протестировано на Ubuntu 22.04).
- Anaconda
- TensorRT SDK
## Установка
Загрузить наборы данных и репозитории:
```shell
source load_proj.sh
```
Поднять окружение:
```shell
source env_setup.sh
```
## Инференс
Скрипт `infer.sh` имеет 3 флага:
- `--ESRGAN`  запускает инференс Real-ESRGAN на наборах данных из `datasets/`. С
- `--RT4KSR`  запускает инференс RT4KSR на наборах данных из `datasets/`.
- `--metrics` запускает подсчёт метрик по результатам инференса.
Cкрипт самостоятельно запускает окружения, установленные через `env_setup.sh` - никакие дополнительные действия совершать не нужно.
```shell
source infer.sh --ESRGAN --RT4KSR --metrics
```
## Результаты
Данные, полученные в результате запуска скрипта `infer.sh` с аргументом `--metrics`, находятся в `study/artifacts/`.