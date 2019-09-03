import os
import librosa
from librosa.display import specshow
import threading
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


OGG_DIR = 'data/raw/DALI_v1.0/ogg_audio_splits/'
SPEC_DIR = 'data/processed/check/'

exitFlag = 0

class myThread (threading.Thread):
    def __init__(self, name, s, e, data):
        threading.Thread.__init__(self)
        self.name = name
        self.s = s
        self.e = e
        self.data = data
    def run(self):
        print ("Starting " + self.name)
        print_time(self.name, self.s, self.e, self.data)
        print ("Exiting " + self.name)

def print_time(threadName, s, e, data):
    if exitFlag:
        threadName.exit()
    #data = os.listdir(OGG_DIR)
    for i in range(s,e):
        try:
            a = data[i]
            y, _ = librosa.load('{}{}'.format(OGG_DIR, a))
            plt.figure(i)
            sa = librosa.feature.melspectrogram(y)
            specshow(librosa.power_to_db(sa, ref=np.max), fmax=8000)
            plt.tight_layout()
            plt.savefig('{}{}.png'.format(SPEC_DIR, a.split('.')[0]), bbox_inches='tight')
            plt.close(i)
            print(i)
        except Exception as e:
            print(e)
            plt.close(i)
            continue

data = os.listdir(OGG_DIR)
for i in range(0,1000):
    a = data[i]
    #plt.figure()
    y, _ = librosa.load('{}{}'.format(OGG_DIR, a))
    plt.figure(i)
    plt.axis('off')
    plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    s = librosa.feature.melspectrogram(y)
    specshow(librosa.power_to_db(s, ref=np.max), fmax=8000)
    plt.tight_layout()
    plt.savefig('{}{}.png'.format(SPEC_DIR, a.split('.')[0]), bbox_inches='tight')
    plt.close(i)
    print(i)
# Create new threads
#thread1 = myThread("Thread-11", 2000, 2005, data)
#thread2 = myThread("Thread-12", 2005, 2010, data)
#thread3 = myThread("Thread-13", 2010, 2015, data)
#thread4 = myThread("Thread-14", 2015, 2020, data)
#thread5 = myThread("Thread-15", 2020, 2025, data)
#thread6 = myThread("Thread-16", 2500, 2600)
#thread7 = myThread("Thread-17", 2600, 2700)
#thread8 = myThread("Thread-18", 2700, 2800)
#thread9 = myThread("Thread-19", 2800, 2900)
#thread10 = myThread("Thread-20", 2900, 3000)
#thread11 = myThread("Thread-21", 3000, 3100)
#thread12 = myThread("Thread-22", 3100, 3200)
#thread13 = myThread("Thread-23", 3200, 3300)
#thread14 = myThread("Thread-24", 3300, 3400)
#thread15 = myThread("Thread-25", 3400, 3500)
#thread16 = myThread("Thread-26", 3500, 3600)
#thread17 = myThread("Thread-27", 3600, 3700)
#thread18 = myThread("Thread-28", 3700, 3800)
#thread19 = myThread("Thread-29", 3800, 3900)
#thread20 = myThread("Thread-30", 3900, 4000)

# Start new Threads
#thread1.start()
#thread2.start()
#thread3.start()
#thread4.start()
#thread5.start()
#thread6.start()
#thread7.start()
#thread8.start()
#thread9.start()
#thread10.start()
#thread11.start()
#thread12.start()
#thread13.start()
#thread14.start()
#thread15.start()
#thread16.start()
#thread17.start()
#thread18.start()
#thread19.start()
#thread20.start()

#thread1.join()
#thread2.join()
#thread3.join()
#thread4.join()
#thread5.join()
#thread6.join()
#thread7.join()
#thread8.join()
#thread9.join()
#thread10.join()
#thread11.join()
#thread12.join()
#thread13.join()
#thread14.join()
#thread15.join()
#thread16.join()
#thread17.join()
#thread18.join()
#thread19.join()
#thread20.join()
print ("Exiting Main Thread")
>>>>>>> Stashed changes
