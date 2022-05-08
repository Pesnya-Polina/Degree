import numpy as np
import pandas as pd
import os
import pickle
import features as f
from typing import Iterable, Set, Sequence, Tuple
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


def get_msku_by_hid(hid: int, path: str, sep: str = '\t') -> Set:
    """
    Получает список msku относящихся к одному хиду.

    Parameters
    ----------
    hid: int
        Номер хида.
    path: str
        Путь, по которому лежит таблица с распределением msku по hid. В таблице должно содержаться поле msku и поле hid,
        со значениями, соответствующим msku.
    sep: str, default='\t'
        Разделитель данных в файле с распределением msku по hid.

    Returns
    -------
    set с msku, относящихся к данному хиду.
    """
    hid_model_msku = pd.read_csv(path, sep=sep)
    msku_s = set(hid_model_msku[hid_model_msku.hid == hid].msku)
    return msku_s


def collect_msku_data(msku: Iterable, path: str, header: bool = True, gmv_ind: int = 0, msku_ind: int = 1,
                      date_ind: int = 2) -> pd.DataFrame:
    """
    Собирает данные по временным рядам, относящимся к данному множеству msku и заполняет пропуски в них.

    Parameters
    ----------
    msku: Iterable
        Набор msku для которых необходимо собрать данные.
    path: str
        Пусть до файла с данными. В каждой строчке файла должна содержаться информация о номере msku, дате и
        gmv (таргет).
    header: bool, default=True
        Есть ли в файле строка с названием полей.
    gmv_ind: int, default=0
        Индекс положения gmv в строке файла.
    msku_ind: int, default=1
        Индекс положения msku в строке файла.
    date_ind: int, default=2
        Индекс положения даты в строке файла.

    Returns
    -------
    pd.DataFrame собранных данных.
    """
    # Данные из файла
    all_data = []
    file = open(path)
    if header:
        file.readline()
    for line in tqdm(file):
        line_arr = line.split()
        gmv = line_arr[gmv_ind]
        ms = int(line_arr[msku_ind])
        date = line_arr[date_ind]
        if ms in msku:
            all_data.append((ms, pd.to_datetime(date), float(gmv)))
    file.close()
    all_data = pd.DataFrame(all_data, columns=['msku', 'ds', 'y'])

    # фактический считанный набор msku
    new_msku = set(all_data.msku)

    # Заполняем пропуски в датах
    arr = []
    start_date = np.min(all_data.ds)
    end_date = np.max(all_data.ds)
    for ms in new_msku:
        cur_data = all_data[all_data.msku == ms]
        cur_data = pd.DataFrame(pd.date_range(start_date, end_date),
                                columns=['ds']).set_index('ds').join(cur_data.set_index('ds')).reset_index()
        cur_data['msku'].fillna(ms, inplace=True)
        cur_data['y'].fillna(0, inplace=True)
        arr.append(cur_data)

    all_data = pd.concat(arr, axis=0).reset_index(drop=True)
    all_data['msku'] = all_data.msku.astype(int)
    return all_data


class MSKUTimeSeriesModel:
    """
    Модель для предсказания поведения нескольких временных рядов из msku, относящихся к одному хиду.

    В модели строится предсказание временного ряда для каждого отдельного msku по дням, используя линейную регрессию,
    которая принимает на вход различные фичи, построенные на окне исторических данных и на информации о дне
    предсказания. Далее (опционально) модель переучивается для каждого отдельного временного ряда, используя в качестве
    входной фичи также предсказание лучшей по метрикам модели на предыдущем шаге.

    Parameters
    ----------
    usd_path: str, default=None
         Путь до файла с историей по курсу доллара. В файле в каждой строке дата и курс, разделённые запятой. Первая
         строка - заголовок. При значении None параметра эта фича в модели использоваться не будет.
    forecast_period: int, default=8
         Количество дней, на которое модель делает предсказание.
    window_width: int, default=30
        Ширина окна, которое используется в построении фичей на исторических данных.
    num_folds: int, default=10
        Количество фолдов на кросс-валидации в обучении.
    with_refit: bool, default=True
        Переобучать ли, с использованием предсказаний лучшей модели в качестве фичи.

    Attributes
    ----------
    models: dict
        Словарь моделей, по ключу msku содержащий список из forecast_period моделей на каждый день предсказания.
    scalers: dict
        Словарь scaler, по ключу msku содержащий scaler данных соответствующей модели.
    first_models: (optional) dict
        При with_refit=True содержит модели до переобучения с использованием предсказания лучшей модели.
    first_scalers: (optional) dict
        При with_refit=True содержит scalers до переобучения с использованием предсказания лучшей модели.
    best_msku: int
        При with_refit=True содержит msku лучшей модели, использующейся для переобучения остальных msku.
    """
    def __init__(self, usd_path: str = None, forecast_period: int = 8, window_width: int = 30, num_folds: int = 10,
                 with_refit: bool = True):
        self.usd_path = usd_path
        if self.usd_path:
            f.set_usd_history(self.usd_path)

        self.forecast_period = forecast_period
        self.window_width = window_width
        self.num_folds = num_folds
        self.with_refit = with_refit

        self.models = dict()
        self.scalers = dict()

        self.first_models = dict()
        self.first_scalers = dict()
        self.best_msku = None

    def fit(self, time_series: pd.DataFrame):
        """
        Выполняет обучение модели.

        Parameters
        ----------
        time_series: pd.DataFrame
            Датасет с данными по временным рядам, содержащий следующие поля:
                msku - номер msku
                ds - дата
                y - таргет (gmv)

        Returns
        -------
        self : object
            Обученная модель
        """
        msku = set(time_series.msku)

        ##### ОБУЧАЕМ МОДЕЛЬ ДЛЯ КАЖДОГО MSKU и сохраняем результат
        if self.with_refit:
            print('Обучение первой модели:')
            self.first_models, self.first_scalers, errors_by_msku = self._train_block(time_series, msku)
        else:
            print('Обучение модели:')
            self.models, self.scalers, errors_by_msku = self._train_block(time_series, msku)

        ##### ПЕРЕОБУЧАЕМ С ИСПОЛЬЗОВАНИЕМ РЕЗУЛЬТАТА ЛУЧШЕЙ МОДЕЛИ
        if self.with_refit:
            # Находим лучшую модель
            self.best_msku = min(errors_by_msku, key=lambda n: n['error'])['msku']
            worst_msku = msku - {self.best_msku}
            best_model = self.first_models[self.best_msku]
            best_scaler = self.first_scalers[self.best_msku]

            # Подготовим набор предсказаний для использования в переобучении.
            # Собираем таблицу, где в каждой строке:
            #   start_data - дата начала интервала для текущего предсказания
            #           (то есть для self.forecast_period предсказаний эта дата одинаковая - дата первого из них),
            #   номер модели (начиная с нуля) текущего предсказания,
            #   само предсказание
            #   прирост
            best_predictions = []
            best_data = time_series[time_series.msku == self.best_msku][['ds', 'y']].reset_index(drop=True)
            for i in range(len(best_data) - self.forecast_period - self.window_width + 1):
                window_features = pd.DataFrame(f.build_window_features(best_data.loc[i:i + self.window_width - 1])).T
                point_features = f.build_single_day_features(
                    best_data.loc[i + self.window_width:i + self.window_width + self.forecast_period - 1])
                date = best_data.loc[i + self.window_width].ds

                x = pd.concat([pd.concat([window_features] * self.forecast_period, axis=0).reset_index(drop=True),
                               point_features], axis=1)
                x = pd.DataFrame(best_scaler.transform(x))

                prev_pred = best_data.loc[i + self.window_width - 1].y
                for j in range(self.forecast_period):
                    pred = best_model[j].predict(x.iloc[[j]])[0]
                    incr = pred - prev_pred
                    prev_pred = pred
                    best_predictions.append([date, j, pred, incr])
            best_predictions = pd.DataFrame(best_predictions,
                                            columns=['start_date', 'model_number', 'prediction', 'increment'])

            print('Переобучение с использованием результата лучшей:')
            self.models, self.scalers, errors_by_msku = self._train_block(time_series, worst_msku, best_predictions)
            self.models[self.best_msku] = self.first_models[self.best_msku]
            self.scalers[self.best_msku] = self.first_scalers[self.best_msku]

    def _train_block(self, time_series: pd.DataFrame, train_msku: Set, best_predictions: pd.DataFrame = None) -> Tuple:
        """
        Главный обучающий блок. Не предполагается к самостоятельному использованию (используется внутри метода fit)

        Parameters
        ----------
        time_series: pd.DataFrame
            Датасет с данными по временным рядам, содержащий следующие поля:
                msku - номер msku
                ds - дата
                y - таргет (gmv)
        train_msku: set
            Набор msku, на которых следует обучаться.
        best_predictions: pd.DataFrame
            Датасет с предсказанием лучшей модели.
        Returns
        -------
        Tuple, состоящий из словаря моделей, словаря scalers и списка пар номер msku-ошибка модели.
        """
        errors_by_msku = []
        models_by_msku = dict()
        scalers_by_msku = dict()
        refit = False
        if best_predictions is not None:
            refit = True

        for cur_msku in tqdm(train_msku):
            data = time_series[time_series.msku == cur_msku][['ds', 'y']].reset_index(drop=True)

            ##### ОБУЧАЕМ SCALER
            window_features = []
            point_features = []
            best_pred_features = []
            for i in range(len(data) - self.forecast_period - self.window_width + 1):
                window_features.append(pd.DataFrame(f.build_window_features(data.loc[i:i + self.window_width - 1])).T)
                point_features.append(f.build_single_day_features(
                    data.loc[i + self.window_width:i + self.window_width + self.forecast_period - 1]))
                if refit:
                    date = data.loc[i + self.window_width].ds
                    best_pred_features.append(best_predictions[(best_predictions.start_date == date) &
                                                               (best_predictions.model_number == 0)]
                                              [['prediction', 'increment']])

            x = pd.concat([pd.concat(window_features, axis=0).reset_index(drop=True),
                           pd.concat([x.loc[[0]] for x in point_features], axis=0).reset_index(drop=True)], axis=1)
            if refit:
                x = pd.concat([x, pd.concat(best_pred_features, axis=0).reset_index(drop=True)], axis=1)
            scaler = StandardScaler()
            scaler.fit(x)

            ##### ОБУЧАЕМ МОДЕЛЬ
            all_models = []
            all_errors = []

            for train, test in self.cross_val_time_series(data):
                n_train = len(train)

                # Формируем фичи трейна
                window_features = []
                point_features = []
                best_pred_features = []
                for i in range(n_train - self.forecast_period - self.window_width + 1):
                    window_features.append(
                        pd.DataFrame(f.build_window_features(train.loc[i:i + self.window_width - 1])).T)
                    point_features.append(f.build_single_day_features(
                        train.loc[i + self.window_width:i + self.window_width + self.forecast_period - 1]))
                    if refit:
                        date = train.loc[i + self.window_width].ds
                        best_pred_features.append(
                            best_predictions[best_predictions.start_date == date].sort_values('model_number')
                            [['prediction', 'increment']].reset_index(drop=True))

                # Обучаем модели
                models = []
                for i in range(self.forecast_period):
                    x = pd.concat([pd.concat(window_features, axis=0).reset_index(drop=True),
                                   pd.concat([x.loc[[i]] for x in point_features], axis=0).reset_index(drop=True)],
                                  axis=1)
                    if refit:
                        x = pd.concat([x, pd.concat([x.loc[[i]] for x in best_pred_features],
                                                    axis=0).reset_index(drop=True)], axis=1)
                    x = scaler.transform(x)
                    y = train.loc[self.window_width + i: n_train - self.forecast_period + i].y

                    m = SGDRegressor()
                    m.fit(x, y)
                    models.append(m)

                # Сохраняем их в общий массив моделей
                all_models.append(models)

                # Формируем фичи теста
                window_features_test = pd.DataFrame(f.build_window_features(train[-self.window_width:])).T
                point_features_test = f.build_single_day_features(test)
                best_pred_features_test = None
                if refit:
                    best_pred_features_test = best_predictions[
                        best_predictions.start_date == test.loc[0].ds].sort_values(
                        'model_number')[['prediction', 'increment']].reset_index(drop=True)

                # Валидируемся
                x = pd.concat([pd.concat([window_features_test] * self.forecast_period, axis=0).reset_index(drop=True),
                               point_features_test], axis=1)
                if refit:
                    x = pd.concat([x, best_pred_features_test], axis=1)
                x = pd.DataFrame(scaler.transform(x))

                y = test.y
                y_pred = [0] * len(y)

                for i in range(len(y)):
                    y_pred[i] = models[i].predict(x.iloc[[i]])
                y_pred = np.array(y_pred)
                error = mean_absolute_error(y, y_pred)

                # добавляем все ошибки в общий массив ошибок
                all_errors.append(error)

            ##### ПОЛУЧАЕМ ИТОГОВУЮ МОДЕЛЬ И ОШИБКУ
            # в качестве итоговой модели возьмём модель с наименьшей ошибкой
            model = all_models[np.argmin(all_errors)]  # модель - это список из моделей по дням предсказания
            errors_by_msku.append({'msku': cur_msku, 'error': np.min(all_errors)})

            models_by_msku[cur_msku] = model
            scalers_by_msku[cur_msku] = scaler

        # Возвращаем словарь с моделями, словарь с scalers и список ошибок
        return models_by_msku, scalers_by_msku, errors_by_msku

    def predict(self, history: pd.DataFrame):
        """
        Получает предсказание модели.

        Parameters
        ----------
        history: pd.DataFrame
            Датасет с историческими данными по всем msku (можно использовать тот же самый, что и для обучения).
            Обязательные поля:
                msku - номер msku
                ds - дата
                y - таргет (gmv)

        Returns
        -------
        C : dict
            Возвращает словарь, где по ключу msku можно получить предсказание на forecast_period дней.
        """
        # стартовая и конечная дата истории
        end_date = np.max(history.ds)
        start_date = end_date - pd.to_timedelta(f'{self.window_width - 1} days')
        msku = set(history.msku)
        train_msku = msku
        if self.with_refit:
            if self.best_msku not in msku:
                raise ValueError('best_msku отсутствует в наборе msku переданном в истории')
            train_msku = train_msku - {self.best_msku}

        predictions = dict()
        # Получаем предсказания лучшей модели
        y_pred_best = None
        if self.with_refit:
            data = history[history.msku == self.best_msku]
            data = pd.DataFrame(pd.date_range(start_date, end_date),
                                columns=['ds']).set_index('ds').join(data.set_index('ds')).reset_index()
            data = data[['ds', 'y']]
            data['y'].fillna(0, inplace=True)

            s = end_date + pd.to_timedelta('1 days')
            e = s + pd.to_timedelta(f'{self.forecast_period - 1} days')
            dates = pd.DataFrame(pd.date_range(s, e), columns=['ds'])

            window_features_test = pd.DataFrame(f.build_window_features(data)).T
            point_features_test = f.build_single_day_features(dates)

            x = pd.concat([pd.concat([window_features_test] * self.forecast_period, axis=0).reset_index(drop=True),
                           point_features_test], axis=1)
            x = pd.DataFrame(self.scalers[self.best_msku].transform(x))

            y_pred = [0] * self.forecast_period
            for i in range(self.forecast_period):
                y_pred[i] = self.models[self.best_msku][i].predict(x.iloc[[i]])
            predictions[self.best_msku] = np.array(y_pred)
            y_pred = pd.DataFrame(y_pred)
            y_pred_incr = f.increment(pd.DataFrame(y_pred), data.y.iloc[0])
            y_pred_best = pd.concat([y_pred, y_pred_incr], axis=1)
            y_pred_best.columns = ['prediction', 'increment']

        # Используя предсказания лучшей модели предсказываем остальные
        for ms in train_msku:
            data = history[history.msku == ms]
            data = pd.DataFrame(pd.date_range(start_date, end_date),
                                columns=['ds']).set_index('ds').join(data.set_index('ds')).reset_index()
            data = data[['ds', 'y']]
            data['y'].fillna(0, inplace=True)

            s = end_date + pd.to_timedelta('1 days')
            e = s + pd.to_timedelta(f'{self.forecast_period - 1} days')
            dates = pd.DataFrame(pd.date_range(s, e), columns=['ds'])

            window_features_test = pd.DataFrame(f.build_window_features(data)).T
            point_features_test = f.build_single_day_features(dates)

            x = pd.concat([pd.concat([window_features_test] * self.forecast_period, axis=0).reset_index(drop=True),
                           point_features_test], axis=1)
            if self.with_refit:
                x = pd.concat([x, y_pred_best], axis=1)
            x = pd.DataFrame(self.scalers[ms].transform(x))

            y_pred = [0] * self.forecast_period

            for i in range(self.forecast_period):
                y_pred[i] = self.models[ms][i].predict(x.iloc[[i]])
            predictions[ms] = np.array(y_pred)

        return predictions

    def train_test_split(self, time_series: pd.DataFrame) -> Tuple:
        """
        Делит выборку на обучающую и тестовую.

        Parameters
        ----------
        time_series: pd.DataFrame
            Датасет с данными по временным рядам, содержащий следующие поля:
                msku - номер msku
                ds - дата
                y - таргет (gmv)

        Returns
        -------
        Tuple из двух pd.DataFrame: обучающего и тестового.
        """
        msku = set(time_series.msku)
        train = []
        test = []
        for ms in msku:
            data = time_series[time_series.msku == ms].reset_index(drop=True)
            train.append(data.iloc[:-self.forecast_period].reset_index(drop=True))
            test.append(data.iloc[-self.forecast_period:].reset_index(drop=True))
        train = pd.concat(train, axis=0).reset_index(drop=True)
        test = pd.concat(test, axis=0).reset_index(drop=True)
        return train, test

    def cross_val_time_series(self, data: Sequence) -> Tuple:
        """
        Генерирует пары: обучающая выборка - валидационная выборка для кросс валидации на временных рядах.

        Parameters
        ----------
        data: Sequence
            Последовательность данных временного ряда.

        Returns
        -------
        Tuple - пара: обучающая выборка - валидационная выборка.
        """
        first_train_len = len(data) - self.forecast_period * self.num_folds
        add = (len(data) - first_train_len) % self.forecast_period
        for i in range(first_train_len + add, len(data) - self.forecast_period + 1, self.forecast_period):
            yield data[:i].reset_index(drop=True), data[i:i + self.forecast_period].reset_index(drop=True)

    def save_models(self, path_to_save: str):
        """
        Сохраняет полученные модели на жёсткий диск.

        Parameters
        ----------
        path_to_save: str
            Путь для сохранения модели. По этому пути будет создана папка model с последней версией модели и папка
            first_model (опционально, при with_refit=True) с версией модели до обучения. В каждой папке по папкам с
            соответствующими номерами msku хранятся forecast_period моделей (на каждый день предсказания) и scaler.
        """
        if path_to_save[-1] == '/':
            path = path_to_save[:-1]
        else:
            path = path_to_save

        with open(f'{path}/window_width', 'x') as file:
            file.write(str(self.window_width))

        os.mkdir(f'{path}/model/')
        for msku in self.models:
            os.mkdir(f'{path}/model/msku_{msku}/')
            with open(f'{path}/model/msku_{msku}/scaler', 'wb') as file:
                pickle.dump(self.scalers[msku], file)
            for i in range(len(self.models[msku])):
                with open(f'{path}/model/msku_{msku}/model_{i}', 'wb') as file:
                    pickle.dump(self.models[msku][i], file)

        if self.with_refit:
            with open(f'{path}/best_msku', 'x') as file:
                file.write(str(self.best_msku))
            os.mkdir(f'{path}/first_model/')
            for msku in self.first_models:
                os.mkdir(f'{path}/first_model/msku_{msku}/')

                with open(f'{path}/first_model/msku_{msku}/scaler', 'wb') as file:
                    pickle.dump(self.scalers[msku], file)
                for i in range(len(self.models[msku])):
                    with open(f'{path}/first_model/msku_{msku}/model_{i}', 'wb') as file:
                        pickle.dump(self.models[msku][i], file)

    def download_models(self, path_to_download: str):
        """
        Считывает сохранённую модель и записывает её в атрибуты класса. Для использования необходимо правильно
        инициализировать модель: передать данные той же валюты.

        Parameters
        ----------
        path_to_download: str
            Путь до сохранённой модели.
        """
        if path_to_download[-1] == '/':
            path = path_to_download[:-1]
        else:
            path = path_to_download

        # Проверяем, переобучалась ли модель
        if len(os.listdir(f'{path}')) == 2:
            self.with_refit = True

        # Получаем длину прогнозируемого интервала
        msku_dir = os.listdir(f'{path}/model')[0]
        self.forecast_period = len(os.listdir(f'{path}/model/{msku_dir}')) - 1

        # Получаем ширину окна
        with open(f'{path}/window_width', 'r') as file:
            self.window_width = int(file.readline())

        # Получаем информацию о лучшей модели, если было переобучение
        if self.with_refit:
            with open(f'{path}/best_msku', 'r') as file:
                self.best_msku = int(file.readline())

        # Скачиваем итоговые модели
        for msku in [int(x[5:]) for x in os.listdir(f'{path}/model')]:
            self.models[msku] = []
            for i in range(self.forecast_period):
                with open(f'{path}/model/msku_{msku}/model_{i}', 'rb') as file:
                    model = pickle.load(file)
                    self.models[msku].append(model)
            with open(f'{path}/model/msku_{msku}/scaler', 'rb') as file:
                self.scalers[msku] = pickle.load(file)

        #  Скачиваем первоначальные модели (если таковые были).
        if self.with_refit:
            for msku in [int(x[5:]) for x in os.listdir(f'{path}/first_model')]:
                self.first_models[msku] = []
                for i in range(self.forecast_period):
                    with open(f'{path}/first_model/msku_{msku}/model_{i}', 'rb') as file:
                        model = pickle.load(file)
                        self.first_models[msku].append(model)
                with open(f'{path}/first_model/msku_{msku}/scaler', 'rb') as file:
                    self.first_models[msku] = pickle.load(file)
