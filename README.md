# Bahavior

* [Dataset research](https://github.com/mxm0312/Generative_models/blob/main/behavior_research.ipynb)

  * При помощи PCA было найдено необходимое минимальное число главных компонент для сохранения `95% информации`. В результате удалось уменьшить признаковоее пространство вредоносного поведения `с 853 до 105`
  * Перед использованием PCA был выбран препроцессинг `MinMaxScaler` для признаков, чтобоы отнормировать их.
  * Результаты из этого ноутбука отражены в классе [BehaviourDataset](https://github.com/mxm0312/Generative_models/blob/main/datasets/behavior.py)
  
 * [Bahavior Classifier](https://github.com/mxm0312/Generative_models/blob/main/bahavior_classifier.ipynb)
    * Исходный датасет был поделен на две части: train и val. Первая использовалась для обучения классификатора, а вторая соответственно для валидации на новых данных
    * Наиболее удачной оказалась архитектура полносвязной нейронной сети. На ней удалось достичь `82% точности` на валидации. Далее функция потерь и точность валидации выходит на плато
    
  # Generative models

Для обученных моделей генерации вредоносного поведения введены 2 метрики: первая - это вероятность, возвращаемая классификатором вредоносного поведения. Вторая - это минимальное расстояние между сгенерированным поведением и реальным, чтобы понять насколько оно уникально и отличается от уже существующего
  | **Model** | **Classifier prob** | **min dist to real sample** |
|:---------:|:-------------------:|:---------------------------:|
|    VAE    |        0.4557       |           14.9270           |
|    GAN    |        0.9483       |           10.9947           |
  


    
