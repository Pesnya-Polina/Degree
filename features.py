import numpy as np
import pandas as pd
import calendar
from typing import List, NoReturn, Iterable

USD_HISTORY = None


class WindowFunctions:
    @staticmethod
    def window_mean(df: pd.Series) -> List:
        """
        Возвращает среднее окна.
        """
        return [np.mean(df)]

    @staticmethod
    def half_mean(df: pd.Series) -> List:
        """
        Возвращает среднее в последней половине окна.
        """
        sep = len(df) // 2
        return [np.mean(df[sep:])]

    @staticmethod
    def quarter_mean(df: pd.Series) -> List:
        """
        Возвращает среднее в последней четверти окна.
        """
        sep = len(df) // 4
        return [np.mean(df[-sep:])]

    @staticmethod
    def window_max(df: pd.Series) -> List:
        """
        Возвращает максимум в окне.
        """
        return [np.max(df)]

    @staticmethod
    def window_min(df: pd.Series) -> List:
        """
        Возвращает минимум в окне.
        """
        return [np.min(df)]

    @staticmethod
    def window_var(df: pd.Series) -> List:
        """
        Возвращает средне квадратичное отклонение в окне.
        """
        return [np.var(df)]

    @staticmethod
    def last_three_point(df: pd.Series) -> List:
        """
        Возвращает последние три значения в окне.
        """
        return list(df[-3:])

    @staticmethod
    def window_mean_incr(df: pd.Series) -> List:
        """
        Возвращает среднее значение прироста в окне.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        return [np.mean(incr)]

    @staticmethod
    def half_mean_incr(df: pd.Series) -> List:
        """
        Возвращает среднее значение прироста в последней половине окна.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        sep = len(incr) // 2
        return [np.mean(incr[sep:])]

    @staticmethod
    def quarter_mean_incr(df: pd.Series) -> List:
        """
        Возвращает среднее значение прироста в последней четверти окна.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        sep = len(incr) // 4
        return [np.mean(incr[sep:])]

    @staticmethod
    def window_max_incr(df: pd.Series) -> List:
        """
        Возвращает максимум прироста в окне.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        return [np.max(incr)]

    @staticmethod
    def window_min_incr(df: pd.Series) -> List:
        """
        Возвращает минимум прироста в окне.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        return [np.min(incr)]

    @staticmethod
    def window_var_incr(df: pd.Series) -> List:
        """
        Возвращает среднее квадратичное отклонение прироста в окне.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        return [np.var(incr)]

    @staticmethod
    def last_three_incr(df: pd.Series) -> List[float]:
        """
        Возвращает последние три значения прироста в окне.
        """
        incr = np.array(df[1:]) - np.array(df[:-1])
        return list(incr[-3:])


class SingleDayFunctions:
    @staticmethod
    def dummy_month(date: pd.Timestamp) -> List[int]:
        """
        Возвращает массив нулей с единицей на позиции нужного месяца.
        """
        res = [0] * 12
        res[date.month - 1] = 1
        return res

    @staticmethod
    def dummy_weekday(date: pd.Timestamp) -> List[int]:
        """
        Возвращает массив нулей с единицей на позиции нужного дня недели (начиная с понедельника).
        """
        res = [0] * 7
        res[date.weekday()] = 1
        return res

    @staticmethod
    def usd_rate(date: pd.Timestamp) -> float:
        """
        Возвращает курс доллара на данную дату. Использует глобальную переменную USD_HISTORY - файл с данными по курсу
        доллара. См. также set_usd_history
        """
        return USD_HISTORY.loc[date].usd


def set_usd_history(path: str, sep: str = ',', start_date: str = '2017-01-01') -> NoReturn:
    """
    Задаёт или меняет значение глобальной переменной USD_HISTORY - файл с данными по курсу доллара. См. также
    PointFunctions.usd_rate

    Parameters
    ----------
    path: str
        Путь, по которому лежит файл с историей курса либо новый файл с историей. Файл должен быть в формате .csv
    sep: str, default=','
        Разделитель данных в файле с историей.
    start_date: str, default='2017-01-01'
        Дата, с которой должна начинаться история.

    Returns
    -------
    None
    """
    global USD_HISTORY

    usd_rub = pd.read_csv(path, sep=sep)
    usd_rub.columns = ['ds', 'usd']
    usd_rub['ds'] = pd.to_datetime(usd_rub.ds)
    usd_rub.sort_values('ds', inplace=True)
    usd_rub.reset_index(drop=True, inplace=True)
    usd_rub.set_index('ds', inplace=True)
    usd_rub = usd_rub.loc[pd.to_datetime(start_date):]

    dates = pd.date_range(usd_rub.index.min(), usd_rub.index.max())
    usd = pd.DataFrame([None] * len(dates), index=dates, columns=['usd'])
    usd.reset_index(inplace=True)
    usd.columns = ['ds', 'usd']
    usd.set_index('ds', inplace=True)
    usd['usd'] = usd_rub[['usd']]
    usd.interpolate(inplace=True)
    USD_HISTORY = usd


def increment(arr: Iterable, start_val: float) -> pd.DataFrame:
    """
    По исходному временному ряду получает ряд приростов (прирост - разница значений текущей даты и предшествующей)

    Parameters
    ----------
    arr: Iterable
        Исходный временной ряд
    start_val: float
        Стартовое значение, не входящее в исходный временной ряд (для вычисления прироста от первого дня ряда)
    Returns
    -------
    pd.DataFrame из ряда приростов
    """
    np_arr = np.array(arr)
    return pd.DataFrame(np_arr - np.insert(np_arr[:-1], 0, start_val).reshape(-1, 1)).reset_index(drop=True)


def build_window_features(series: pd.DataFrame) -> pd.Series:
    """
    Собирает набор оконных фичей для заданного окна. В фичи входит:
    1. Среднее в окне
    2. Среднее во второй половине окна
    3. Среднее в последней четверти окна
    4. Максимум в окне
    5. Минимум в окне
    6. Дисперсия в окне
    7. Последние три точки в окне
    8. Среднее значение прироста в окне
    9. Среднее значение прироста во второй половине окна
    10. Среднее значение прироста в последней четверти окна
    11. Максимум прироста в окне
    12. Минимум прироста в окне
    13. Дисперсия прироста в окне
    14. Последнее три значения прироста

    Parameters
    ----------
    series: pd.DataFrame
        Содержит окно с данными, для которого нужно построить оконные фичи. Обязательные поля: ds - дата,
        y - таргет (gmv)

    Returns
    -------
    pd.Series с собранными фичами для данного окна. В качестве индексов - названия фичи.
    """
    target = np.array(series.y)
    func_names = [method for method in dir(WindowFunctions) if not method.startswith('__')]
    features = []
    names = []
    for func_name in func_names:
        res = eval('WindowFunctions.' + func_name)(target)
        if len(res) > 1:
            name = [func_name + '_' + str(i) for i in range(len(res))]
        else:
            name = [func_name]
        names.extend(name)
        features.extend(res)
    return pd.Series(features, index=names)


def build_single_day_features(points: pd.DataFrame) -> pd.DataFrame:
    """
    Собирает набор точечных фичей для отдельной даты. В фичи входит:
    1. One hot encoding месяц
    2. One hot encoding день недели
    3. Курс доллара на выбранную дату (если if_usd = True)
    Использует .csv файл с историей курса

    Parameters
    ----------
    points: pd.DataFrame
        Содержит даты, на которые будет строиться предсказание. Должен содержать поле ds - дата.

    Returns
    -------
    pd.DataFrame с собранными фичами на каждую дату. Отдельное поля - отдельная фича, строки - даты.
    """
    res = []
    names = []
    for i in range(len(points)):
        res.append([])

    # Месяц, день недели и курс доллара
    names.extend(list(calendar.month_abbr)[1:])
    names.extend(list(calendar.day_abbr))
    if USD_HISTORY is not None:
        names.append('usd')
    for i, d in enumerate(points.ds.values):
        date = pd.to_datetime(d)
        res[i].extend(SingleDayFunctions.dummy_month(date))
        res[i].extend(SingleDayFunctions.dummy_weekday(date))
        if USD_HISTORY is not None:
            res[i].append(SingleDayFunctions.usd_rate(date))

    return pd.DataFrame(res, columns=names)
