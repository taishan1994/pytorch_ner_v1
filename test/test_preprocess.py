import sys
sys.path.append('..')
from utils import commonUtils
import pickle

data_path = '../data/cner/nor_data'
# train_features, train_callback_info = commonUtils.read_pkl(data_path, 'train')

train_features, train_callback_info = pickle.load(open('../data/cner/nor_data/train.pkl','rb'))
