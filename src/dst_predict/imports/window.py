import numpy as np

def slice(data, indx, size):
    return data[indx:indx + size]

def rolling(records, key, indx, size=64):
    data = slice(records["data"], indx, size)
    return np.array([d[key] for d in data])

def format_predict_data(dst_nT, time_encodings):
    return {
        "dst_nT": dst_nT,
        "time_enc": time_encodings,
    }

# Encode the time of day (hour) as a vector
def encode_hour(timestamps):
    hours = np.array([t.hour for t in timestamps])
    
    sin_hour = np.sin(2 * np.pi * hours / 24)
    cos_hour = np.cos(2 * np.pi * hours / 24)
    
    return np.column_stack([sin_hour, cos_hour])

# Encode the date of year as a vector
def encode_day_of_year(timestamps):
    doy = np.array([t.timetuple().tm_yday for t in timestamps])
    
    sin_doy = np.sin(2 * np.pi * doy / 365.25)
    cos_doy = np.cos(2 * np.pi * doy / 365.25)
    
    return np.column_stack([sin_doy, cos_doy])

# Encode solar rotation with UNIX timestamp as the reference as a vector
def encode_solar_rotation(timestamps):
    seconds = np.array([t.timestamp() for t in timestamps])
    days = seconds / 86400.0
    period = 27.0
    sin_rot = np.sin(2 * np.pi * days / period)
    cos_rot = np.cos(2 * np.pi * days / period)
    return np.column_stack([sin_rot, cos_rot])

# Encode all of the time stamps
def encode_timestamps(timestamps):
    hour = encode_hour(timestamps)
    doy = encode_day_of_year(timestamps)
    solar = encode_solar_rotation(timestamps)
    return np.column_stack([hour, doy, solar])

def _check_bounds(name, rec, indx, size, pred):
    if indx < 0:
        raise ValueError(f"{name}: index cannot be negative integer.")
    if indx >= rec["data"].size - (size + pred):
        raise ValueError(f"{name}: sliding window gone out of bounds.")

def predict(rec, indx, size=64, pred=6):

    indx_abs = indx + size + pred

    _check_bounds("predict", rec, indx, size, pred)

    def inputs(i):
        timestamps = slice(rolling (
            rec, "timestamp", indx_abs - size - pred, size + pred
        ), i, size)
        return format_predict_data (
            slice(rolling (
                rec, "dst_nT", indx_abs - size - pred, size + pred
            ), i, size),
            encode_timestamps(timestamps)
        )
    
    return np.array([ inputs(i) for i in range(0, pred) ])

def training(rec, indx: int, size=64, pred=6):

    indx_abs = indx + size + pred

    _check_bounds("training", rec, indx, size + pred, pred)

    def truths():
        timestamps = rolling(rec, "timestamp", indx_abs + 1, pred)
        return format_predict_data (
            rolling(rec, "dst_nT", indx_abs + 1, pred),
            encode_timestamps(timestamps)
        )

    return {
        "inputs": predict(rec, indx, size, pred),
        "truths": truths()
    }
