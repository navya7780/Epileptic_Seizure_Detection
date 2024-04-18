from django.shortcuts import render, HttpResponse

from django.contrib import messages
from finalproject import settings
import matplotlib.pyplot as plt
import matplotlib
from .utils import get_plot,get_beta_plot,get_alpha_plot,get_theta_plot,get_delta_plot,read,get_acc
from myapp.models import user
from myapp import pyeeg
import mne, keras

import pandas as pd



def login(req):
    try:
        if req.method == "POST":
            file = req.FILES['filewav']
            name = req.POST['name']
            user.objects.create(name = name, media = file)
            return render(req, 'myapp/eegstart.html')
    except Exception:
        messages.warning(req, 'Please enter valid details!!!.....')
        return render(req, 'myapp/mylogin.html')
    
    return render(req, 'myapp/mylogin.html')

def start(req):
    return render(req, 'myapp/eegstart.html')

def fivebands(req):
    return render(req, 'myapp/fivebands.html')

def choose(req):
    try:
        fi = user.objects.last()
        data = read(fi.media)
        chart = get_plot(data)
        return render(req, 'myapp/choose.html', {'chart':chart, 'data1':data})
    except Exception:
        messages.warning(req, 'Error in displaying!!!.....')
        return render(req, 'myapp/eegstart.html')
    
def alpha(req):
    try:
        fi = user.objects.last()
        data = read(fi.media)
        chart = get_alpha_plot(data)
        return render(req, 'myapp/alpha.html', {'chart':chart, 'name':'alpha'})
    except Exception:
        messages.warning(req, 'Error in displaying!!!.....')
        return render(req, 'myapp/fivebands.html')

def theta(req):
    try:
        fi = user.objects.last()
        data = read(fi.media)
        chart = get_theta_plot(data)
        return render(req, 'myapp/alpha.html', {'chart':chart, 'name':'theta'})
    except Exception:
        messages.warning(req, 'Error in displaying!!!.....')
        return render(req, 'myapp/fivebands.html')

def delta(req):
    try:
        fi = user.objects.last()
        data = read(fi.media)
        chart = get_delta_plot(data)
        return render(req, 'myapp/alpha.html', {'chart':chart, 'name':'delta'})
    except Exception:
        messages.warning(req, 'Error in displaying!!!.....')
        return render(req, 'myapp/fivebands.html')

def beta(req):
    try:
        fi = user.objects.last()
        data = read(fi.media)
        chart = get_beta_plot(data)
        return render(req, 'myapp/alpha.html', {'chart':chart, 'name':'beta'})
    except Exception:
        messages.warning(req, 'Error in displaying!!!.....')
        return render(req, 'myapp/fivebands.html')

def preprocess(req):
    try:
        fi = user.objects.last()
        raw = read(fi.media)
        raw1 = raw.load_data().filter(l_freq = 0.25, h_freq = 25)
        chart = get_plot(raw1)
        return render(req, 'myapp/preprocess.html', {'chart':chart, 'data1':raw})
    except Exception:
        messages.warning(req, 'preprocessing unsuccessful!!!.....')
        return render(req, 'myapp/eegstart.html')

def features(req):
    try:
        fi = user.objects.last()
        file = read(fi.media)
        import os
        import glob
        import mne 
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.stats import kurtosis, skew
        from scipy.signal import argrelextrema
        from scipy.signal import welch
        from scipy.integrate import cumtrapz
        import statistics
        import time
        def eeg_features(data):
            data = np.asarray(data)
            res = np.zeros([18])
            Kmax = 5
            Band = [1, 5, 10, 15, 20, 25]
            Fs = 256
            power, power_ratio = pyeeg.bin_power(data, Band, Fs)
            f, P = welch(data, fs=Fs, window='hann',nfft=int(256.0), noverlap=0)

            area_freq = cumtrapz(P, f, initial=0)
            res[0] = np.sqrt(np.sum(np.power(data, 2)) / data.shape[0])
            res[1] = statistics.stdev(data) ** 2
            res[2] = kurtosis(data)
            res[3] = skew(data)
            res[4] = max(data)
            res[5] = min(data)
            res[6] = len(argrelextrema(data, np.greater)[0])
            res[7] = ((data[:-1] * data[1:]) < 0).sum()
            res[8] = pyeeg.hurst(data)
            res[9] = pyeeg.spectral_entropy(data, Band, Fs, Power_Ratio=power_ratio)
            res[10] = area_freq[-1]
            res[11] = f[np.where(area_freq >= res[10] / 2)[0][0]]
            res[12] = f[np.argmax(P)]
            res[13], res[14] = pyeeg.hjorth(data)
            res[15] = power_ratio[0]
            res[16] = power_ratio[1]
            res[17] = power_ratio[2]
            return (res)
        def eeg_preprocessing(file, epoch_length = 5, step_size = 5, start_time = 0):
            start = time.time()
            raw = file
            raw = raw.load_data().filter(l_freq = 0.25, h_freq = 25)
            print(raw.ch_names)
            l = ['T8-P8-0', 'F7-T7', 'FP1-F3', 'FZ-CZ', 'CZ-PZ', 'C4-P4', 'F3-C3', 'FT9-FT10', 'FP2-F4']
            c = 0
            for i in range(0, len(l)):
                if l[i] in raw.ch_names:
                    print(c)
                    c = c + 1
                else:
                    print(l[i])
            if c == len(l):
                raw.pick_channels(ch_names = l)
            channels = raw.ch_names
            res = []
            while start_time <= max(raw.times) + 0.01 - epoch_length:
                features = []
                start, stop = raw.time_as_index([start_time, start_time + epoch_length])
                temp = raw[:, start:stop][0]
                for i in range(0, len(channels)):
                    features.extend(eeg_features(temp[i]).tolist())
                res.append(features)
                print(len(res[0]))
                start_time += step_size
                print("Section ", str(len(res)), "; start: ", start, " ; stop: ", stop)

            feature_names = ["rms", "variance", "kurtosis", "skewness", "max_amp", "min_amp", "n_peaks", "n_crossings", 
			"hurst_exp", "spectral_entropy", "total_power", "median_freq", "peak_freq", 
			"hjorth_mobility", "hjorth_complexity", "power_1hz", "power_5hz", "power_10hz"]
            column_names = []
            for channel in channels:
                for name in feature_names:
                    column_names.append(channel + "_" + name)
            print(len(res[0]))
            res = pd.DataFrame(res, columns = column_names)
            end = time.time()
            print("Finished preprocessing ", file, "took", end - start, "seconds")
            return res
        res = eeg_preprocessing(file)
        res.to_csv(os.path.join('myapp/static/upload/', 'extracted_data' + '.csv'), encoding='utf-8', index=False)
        print("COMPLETED PROCESSING FILE")
        return render(req, 'myapp/features.html', {'sections':len(res)})
    except Exception as e:
        messages.warning(req, 'feature extraction unsuccessful!!!.....')
        print('Exception : ', e)
        return render(req, 'myapp/eegstart.html')

def navbar(req):
    return render(req, 'myapp/navbar.html')

def classify(req, jm):
    import pandas as pd 
    import pickle
    import numpy as np
    data = pd.read_csv('myapp/static/upload/extracted_data.csv')
    data = data.iloc[:, :162]
    
    if jm == 'xgboost':
        name = 'xgboost'
        model = pickle.load(open('xgbcmodel.pkl', 'rb'))
        a = model.predict(data)
    elif jm == 'cnn':
        name = 'CNN'
        new_model = keras.models.load_model('cnn_model.h5')
        d = np.expand_dims(data, axis = 2)
        a = np.argmax(new_model.predict(d), axis = 1)
    elif jm == 'rf':
        name = 'Random forest'
        model = pickle.load(open('randomforestmodel.pkl', 'rb'))
        a = model.predict(data)
    pred = np.max(a)
    res = ''
    if pred == 1:
        res = 'Ictal'
    elif pred == 2:
        res = 'pre-ictal'
    elif pred == 0:
        res = 'normal'
    d = {'1':'Ictal', '2':'pre-ictal', '0':'normal'}
    a = [str(i) for i in a]
    for i in range(0, len(a)):
        a[i] = str(d[a[i]])
    print(a)
    l = []
    l1 = []
    a1 = ['Ictal', 'pre-ictal', 'normal']
    for i in a1:
        l.append(a.count(i))
    print(l)
    def mode(List):
        return max(set(List), key = List.count)
    print(mode(a))
    c = zip(a1, l)
    if l[0] > 1 or l[1] > 1:
        dis = 'Abnormal, Seizure signals are observed'
    else:
        dis = 'Normal'
    return render(req, 'myapp/classify.html', {'res':res, 'model':name, 'count':c, 'classify':dis})

def result(req):
    import pandas as pd
    import pickle
    import numpy as np
    data = pd.read_csv('myapp/static/upload/extracted_data.csv')
    data = data.iloc[:, :162]
    model = pickle.load(open('xgbcmodel.pkl', 'rb'))
    a = model.predict(data)
    d = {'1':'Ictal', '2':'pre-ictal', '0':'normal'}
    a = [str(i) for i in a]
    for i in range(0, len(a)):
        a[i] = str(d[a[i]])
    print(a)
    l = []
    a1 = ['Ictal', 'pre-ictal', 'normal']
    for i in a1:
        l.append(a.count(i))
    print(l)
    c = zip(a1, l)
    if l[0]> 1 or l[1]>1:
        dis = 'Abnormal, Seizure signals are observed'
    else:
        dis = 'Normal'
    chart = get_acc()
    return render(req, 'myapp/result.html', {'count':c, 'classify':dis, 'chart':chart})

def delete(req):
    try:
        d = user.objects.all()
        if d:
            d.delete()
            messages.success(req, 'EXISTING FILES DELETED SUCCESSFULLY..')
            return render(req, 'myapp/eegstart.html')
        else:
            messages.warning(req, 'Files are not deleted!!!........')
            return render(req, 'myapp/eegstart.html')
    except Exception:
        messages.warning(req, 'Files are not deleted!!!.....')
        return render(req, 'myapp/eegstart.html')