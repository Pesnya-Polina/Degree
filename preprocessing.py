import numpy as np
import pandas as pd
import sys


def process_time_series_for_hid(key, rows):
    """
    Отрезает шум в начале временного ряда, или возвращает пустой генератор,
    если ряд имеет слишком много пропусков. Работает с рядами для хидов

    Parameters
    ----------
    rows: list of AttrDict
        Список объектов, с обязательными атрибутами date и value
    """
    SEP_TO_DELETE = 0.55  # задаёт границу метрики holes_frac, выше которой ряды отбрасываются
    PART_OF_MEAN_FOR_SEP = 1 / 5
    LEN_OF_WINDOW = 30
    MIN_PART_OF_SERIES_TO_KEEP = 1 / 6
    MIN_LEN_OF_SERIES_TO_KEEP = 3
    STRIDE = 3

    df = list(rows)

    data = pd.DataFrame([(pd.to_datetime(x.order_date), x.gmv) for x in df], columns=['ds', 'y']).set_index('ds')
    date_range = pd.DataFrame(pd.date_range(start=min(data.index), end=max(data.index)), columns=['ds'])
    date_range = date_range.set_index('ds')
    date_range['y'] = data.y
    if len(date_range) < LEN_OF_WINDOW:
        for i in df:
            yield {
                'hid': key,
                'order_date': i.order_date,
                'gmv': i.gmv
            }

    else:
        mean_in_windows = []
        windows_limits = []
        for date in date_range.iloc[:-LEN_OF_WINDOW - 1:STRIDE].index:
            mean_in_windows.append(np.sum(date_range[date: date + pd.DateOffset(LEN_OF_WINDOW)].y) / LEN_OF_WINDOW)
            windows_limits.append({'start': date, 'end': date + pd.DateOffset(LEN_OF_WINDOW)})

        mean = np.array(mean_in_windows).mean()
        sep_window = 0
        if_sep_found = False
        for ind, val in list(enumerate(mean_in_windows[:-int(len(mean_in_windows) * MIN_PART_OF_SERIES_TO_KEEP)]))[
                        ::-1]:
            if val < mean * PART_OF_MEAN_FOR_SEP:
                if_sep_found = True
                sep_window = ind
                break
        if if_sep_found and (sep_window >= 0):
            sep = windows_limits[sep_window]['end']
        else:
            sep = min(data.index)

        holes_frac = np.sum(date_range.y.isna()) * 1.0 / date_range.shape[0]
        if (holes_frac <= SEP_TO_DELETE) and (len(df) >= MIN_LEN_OF_SERIES_TO_KEEP):
            for x in df[-len(data[sep:]):]:
                yield {
                    'hid': key,
                    'order_date': x.order_date,
                    'gmv': x.gmv
                }


def process_time_series_for_model(key, rows):
    """
    Отрезает шум в начале временного ряда, или возвращает пустой генератор,
    если ряд имеет слишком много пропусков. Работает с рядами для хидов

    Parameters
    ----------
    rows: list of AttrDict
        Список объектов, с обязательными атрибутами date и value
    """
    SEP_TO_DELETE = 0.97  # задаёт границу метрики holes_frac, выше которой ряды отбрасываются
    PART_OF_MEAN_FOR_SEP = 1 / 5
    LEN_OF_WINDOW = 30
    MIN_PART_OF_SERIES_TO_KEEP = 1 / 6
    MIN_LEN_OF_SERIES_TO_KEEP = 3
    STRIDE = 3

    df = [x for x in rows]

    data = pd.DataFrame([(pd.to_datetime(x.order_date), x.gmv) for x in df], columns=['ds', 'y']).set_index('ds')
    date_range = pd.DataFrame(pd.date_range(start=min(data.index), end=max(data.index)), columns=['ds'])
    date_range = date_range.set_index('ds')
    date_range['y'] = data.y
    if len(date_range) < LEN_OF_WINDOW:
        for i in df:
            yield {
                'model_id': key,
                'order_date': i.order_date,
                'gmv': i.gmv
            }

    else:
        mean_in_windows = []
        windows_limits = []
        for date in date_range.iloc[:-LEN_OF_WINDOW - 1:STRIDE].index:
            mean_in_windows.append(np.sum(date_range[date: date + pd.DateOffset(LEN_OF_WINDOW)].y) / LEN_OF_WINDOW)
            windows_limits.append({'start': date, 'end': date + pd.DateOffset(LEN_OF_WINDOW)})

        mean = np.array(mean_in_windows).mean()
        sep_window = 0
        if_sep_found = False
        for ind, val in list(enumerate(mean_in_windows[:-int(len(mean_in_windows) * MIN_PART_OF_SERIES_TO_KEEP)]))[
                        ::-1]:
            if val < mean * PART_OF_MEAN_FOR_SEP:
                if_sep_found = True
                sep_window = ind
                break
        if if_sep_found and (sep_window >= 0):
            sep = windows_limits[sep_window]['end']
        else:
            sep = min(data.index)

        holes_frac = np.sum(date_range.y.isna()) * 1.0 / date_range.shape[0]
        if (holes_frac <= SEP_TO_DELETE) and (len(df) >= MIN_LEN_OF_SERIES_TO_KEEP):
            try:
                for x in df[-len(data[sep:]):]:
                    yield {
                        'model_id': key,
                        'order_date': x.order_date,
                        'gmv': x.gmv
                    }
            except:
                sys.stderr.write(str(sep) + '\n')
                sys.stderr.write(data.to_csv() + '\n')
                raise


def process_time_series_for_msku(key, rows):
    """
    Отрезает шум в начале временного ряда, или возвращает пустой генератор,
    если ряд имеет слишком много пропусков. Работает с рядами для хидов

    Parameters
    ----------
    rows: list of AttrDict
        Список объектов, с обязательными атрибутами date и value
    """
    SEP_TO_DELETE = 0.97  # задаёт границу метрики holes_frac, выше которой ряды отбрасываются
    PART_OF_MEAN_FOR_SEP = 1 / 5
    LEN_OF_WINDOW = 30
    MIN_PART_OF_SERIES_TO_KEEP = 1 / 6
    MIN_LEN_OF_SERIES_TO_KEEP = 3
    STRIDE = 3

    df = [x for x in rows]

    data = pd.DataFrame([(pd.to_datetime(x.order_date), x.gmv) for x in df], columns=['ds', 'y']).set_index('ds')
    date_range = pd.DataFrame(pd.date_range(start=min(data.index), end=max(data.index)), columns=['ds'])
    date_range = date_range.set_index('ds')
    date_range['y'] = data.y
    if len(date_range) < LEN_OF_WINDOW:
        for i in df:
            yield {
                'msku': key,
                'order_date': i.order_date,
                'gmv': i.gmv
            }

    else:
        mean_in_windows = []
        windows_limits = []
        for date in date_range.iloc[:-LEN_OF_WINDOW - 1:STRIDE].index:
            mean_in_windows.append(np.sum(date_range[date: date + pd.DateOffset(LEN_OF_WINDOW)].y) / LEN_OF_WINDOW)
            windows_limits.append({'start': date, 'end': date + pd.DateOffset(LEN_OF_WINDOW)})

        mean = np.array(mean_in_windows).mean()
        sep_window = 0
        if_sep_found = False
        for ind, val in list(enumerate(mean_in_windows[:-int(len(mean_in_windows) * MIN_PART_OF_SERIES_TO_KEEP)]))[
                        ::-1]:
            if val < mean * PART_OF_MEAN_FOR_SEP:
                if_sep_found = True
                sep_window = ind
                break
        if if_sep_found and (sep_window >= 0):
            sep = windows_limits[sep_window]['end']
        else:
            sep = min(data.index)

        holes_frac = np.sum(date_range.y.isna()) * 1.0 / date_range.shape[0]
        if (holes_frac <= SEP_TO_DELETE) and (len(df) >= MIN_LEN_OF_SERIES_TO_KEEP):
            try:
                for x in df[-len(data[sep:]):]:
                    yield {
                        'msku': key,
                        'order_date': x.order_date,
                        'gmv': x.gmv
                    }
            except:
                sys.stderr.write(str(sep) + '\n')
                sys.stderr.write(data.to_csv() + '\n')
                raise