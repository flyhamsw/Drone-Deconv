import numpy as np
import cv2
import csv
from matplotlib import pyplot as plt
from tqdm import tqdm

def create_recall_precision(interest_ch):
    INTEREST_CH = interest_ch

    with open('recall-precision_ch_%d.csv' % INTEREST_CH, 'w') as f:
        for t in tqdm(np.arange(0.05, 1, 0.05)):
            n_TP_sum = 0
            n_FP_sum = 0
            n_FN_sum = 0
            
            for i in range(0, 28000):
                ground_truth = cv2.imread('prediction_result/%d_y.png' % i)[:,:,INTEREST_CH] / 255
                ground_truth = ground_truth == 1
                prediction = cv2.imread('prediction_result/%d_p.png' % i)[:,:,INTEREST_CH] / 255
                prediction = prediction > t
                
                n_TP_sum = n_TP_sum + np.sum((ground_truth == True) & (prediction == True))
                n_FP_sum = n_FP_sum + np.sum((ground_truth == False) & (prediction == True))
                n_FN_sum = n_FN_sum + np.sum((ground_truth == True) & (prediction == False))
                
            recall = n_TP_sum / (n_TP_sum + n_FN_sum)
            precision = n_TP_sum / (n_TP_sum + n_FP_sum)
            
            line = '%f,%f,%f\n' % (t, recall, precision)
            print(line)
            f.write(line)

def parse_csv(fname):
    recall = []
    precision = []
    
    with open(fname, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            recall.append(line[1])
            precision.append(line[2])
            
    return recall, precision
    
def plot_rp_curve(fname, name):
    recall, precision = parse_csv(fname)
    plt.plot(precision, recall, label=name)
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    plt.savefig('rp_curve_%s.png' % name)
    
if __name__=='__main__':
    #create_recall_precision(0)
    create_recall_precision(1)
    
    #plot_rp_curve('recall-precision_ch_0.csv', 'Building')
    #plot_rp_curve('recall-precision_ch_1.csv', 'Road')