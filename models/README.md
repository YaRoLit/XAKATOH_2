В данной папке находятся файлы моделей и предобученных Simple Imputers SKLearn.

Simple Imputers были обучены на статистиках исходного датасета, который использовался для подготовки моделей МЛ. Они используются для заполнения пропущенных значений в конвейере предобработчика поступающих на сервер fastapi запросов.

Часть моделей BankA-decision...BankE-decision - это модели МЛ в формате pickle, вокруг которых построена инфраструктура данного приложения на базе fastapi и которые используются расчета вероятности одобрения/ не одобрения тем или иным банком решения о выдаче кредита по переданным параметрам клиента.

Часть моделей, таких как education.cls, Loan_amount.reg также являются предобученными моделями на основе библиотеки Catboost, но используются во вспомогательных целях. С их помощью (по аналогии с Simple Imputers) происходит заполнение пропущенных значений в получаемых пакетах данных с параметрами клиентов. Так же, как Simple Imputers, они были обучены на исходном датасете, поэтому могут качественно улавливать взаимосвязь между всеми входными признаками для заполнения пропусков.

Модели, используемые для классификации, имеют расширение *.cls. Регрессионные модели - *.reg.

Модели, которые использовались ранее (в предыдущей версии приложения) перемещаются в папку (old).
