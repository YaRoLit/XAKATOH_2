[![Tests](.github/workflows/python-app.yml/badge.svg?event=push)](.github/workflows/python-app.yml)

# ХАКАТОН УрФУ 12/2023. Проект от заказчика <br>["Платежная система Mandarin"](https://mandarin.io/ru)
Цель проекта: создание и развертывание приложения на основе ML-моделей кредитного скоринга, обученных на предоставленном заказчиком наборе данных, для определения вероятности получения клиентами платежной системы кредита в банках в целях повышения конверсии бизнеса.

[1. Описание работы приложения](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Краткое-описание)

[2. Работа с приложением](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Использование-приложения)

[2.1. Приложение в облаке](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Способ-1-(самый-простой):-использование-развернутого-приложения)

[2.2. Развёртывание локально](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Способ-2:-запуск-на-локальном-сервере)

[3. Структура репозитория, описание работы модулей](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Подробное-описание-структуры-репозитория)

[4. Авторы](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md#Команда)

## Краткое описание

Приложение создано на основе веб-фреймворка *FastAPI*, с помощью которого реализуется клиент-серверный интерфейс передачи POST-запросов с клиентскими данными и получения предсказаний моделей о вероятностях одобрения решения о предоставлении кредитов пятью исследуемыми банками (*Bank_A...Bank_E*).

## Использование приложения

### Способ 1 (самый простой): использование развернутого приложения

На период проведения Хакатона приложение развёрнуто на [виртуальной машине в Яндекс облаке](http://158.160.135.101:5000/). Слайд с описанием JSON структуры POST-запроса для получения предсказания моделей находится на странице [/info/](http://158.160.135.101:5000/info/). API для направления POST-запросов с данными находится на странице [/AskOraqul/](http://158.160.135.101:5000/AskOraqul/).

На странице проекта расположен файл Jupyter notebook с [пользовательским интерфейсом отправки POST-запросов на сервер](https://github.com/YaRoLit/XAKATOH_2/blob/main/UI_tests.ipynb), в котором описаны несколько различных вариантов запросов (единичный запрос и отправка серии запросов по табличным данным).

### Способ 2: запуск на локальном сервере

Клонируйте репозиторий:
```
$ git clone git@github.com:YaRoLit/XAKATOH_2.git
```
Установите необходимые для работы приложения библиотеки python:
```
$ cd XAKATOH_2 && pip install -r requirements.txt 
```
Запустите приложение:
```
$ python3 fastapi_app.py
```
На странице проекта расположен файл Jupyter notebook с [пользовательским интерфейсом отправки POST-запросов на сервер](https://github.com/YaRoLit/XAKATOH_2/blob/main/UI_tests.ipynb), в котором описаны несколько различных вариантов запросов (единичный запрос и отправка серии запросов по табличным данным). Для отправки тестовых запросов на локальный сервер, необходимо указать адрес локального сервера (по умолчанию 127.0.0.1:5000). 

## Подробное описание структуры репозитория

Репозиторий содержит следующие папки и файлы. Основные файлы:

- Основной скрипт развертывания сервера для работы приложения [fastapi_app.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/fastapi_app.py). Необходим для запуска приложения, все остальные находящиеся в репозитории скрипты являются его вспомогательными модулями.

- Модуль предобработчика POST-запросов, фильтрации поступивших данных и их преобразования в Pandas dataframe [request_checker.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/request_checker.py).

- Модуль c четырехступенчатым конвейером заполнения пропущенных значений в поступившем фрейме данных [nans_filler.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/nans_filler.py).

- Модуль конвейера создания новых признаков, необходимых для работы модели [features_creator.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/features_creator.py). В репозитории расположено несколько конвейеров от разных участников проекта, актуальным является тот, который не имеет обозначения имени участника в названии файла.

- Модуль предобработчика признака Position от разных авторов (могут подключаться разные варианты в зависимости от используемого конвейера создания новых признаков) [preproc_position.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/preproc_position.py).

Вспомогательные файлы:

- Jupyter notebook с простым пользовательским интерфейсом создания и отправки POST-запросов на сервер, получения ответа от моделей ML. [UI_tests.ipynb](https://github.com/YaRoLit/XAKATOH_2/blob/main/UI_tests.ipynb).

- Интеграционные тесты приложения, встроенные в рабочее окружение репозитория [test_fastapi_app.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/test_fastapi_app.py).

- Тестер качества заполнения пропусков при использовании различных стратегий [nansfiller_tester.py](https://github.com/YaRoLit/XAKATOH_2/blob/main/nansfiller_tester.py).

- Каталог с Jupyter notebooks для создания простых моделей ML, используемых в конвейере заполнения пропусков и для тестирования работоспособности API. Общая структура каталога описана в файле [README.md](https://github.com/YaRoLit/XAKATOH_2/blob/main/service_nb/README.md).

- Каталог с изображениями (лого команды и информационный слайд) используемыми при работе сервера fastapi при обращении к соответствующим страницам. [README.md](https://github.com/YaRoLit/XAKATOH_2/blob/main/images/README.md).

- Каталог с данными, которые использовались при обучении моделей ML и используются при тестировании работы приложения. [README.md](https://github.com/YaRoLit/XAKATOH_2/blob/main/images/README.md).

В указанных файлах приведено подробное описание их работы в целом, а также отдельных функций и блоков кода.  

## Команда
<p align="center">
<img src = './images/logo.png' alt = 'Team logo' align='center'/>
</p>

- T: Татьяна Маухедтинова ([Tatiana302](https://github.com/Tatiana302)): подготовка [readme](https://github.com/YaRoLit/XAKATOH_2/blob/main/README.md), помощь в EDA.
- A: Андрей Крупский ([KrupskiiAndrei](https://github.com/KrupskiiAndrei)): проектный менеджмент, подготовка [проекта решения](https://docs.google.com/document/d/1ftHe8Kgonay7vcks1CiJXJk4O8QZuT4i/edit?usp=sharing&ouid=116001960258803646275&rtpof=true&sd=true), креативные решения.
- I: Игорь Пластов ([chetverovod](https://github.com/chetverovod)): создания [конвейера обработки данных]((https://github.com/YaRoLit/XAKATOH_2/blob/main/Data_preproc/preproc_position.py)), помощь разработке API, обучение моделей.
- S: Станислав Вдовин ([Stasvdovin](https://github.com/Stasvdovin)): помощь в обработке данных и обучении моделей.
- A: Анна Гулич ([Illania](https://github.com/Illania)): Общее руководство командой, EDA, выбор и создание моделей ML.
- Y: Ярослав Литаврин ([YaRoLit](https://github.com/yarolit)): создание [API](https://github.com/YaRoLit/XAKATOH_2.git) для работы моделей на основе fastapi.
