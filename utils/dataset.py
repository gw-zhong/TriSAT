import os
# import cPickle as pickle
import pickle
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

AUDIO = 'covarep'
VISUAL = 'facet'
TEXT = 'glove'
LABEL = 'label'

AUDIO_ = 'audio'
VISUAL_ = 'vision'
TEXT_ = 'text'
LABEL_ = 'labels'

BERT = 'text_bert'
BERT_LABEL = 'regression_labels'


def load_iemocap(data_dir):
    class IEMOCAP(Dataset):
        '''
        PyTorch Dataset for IEMOCAP, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [
                self.audio[idx, :, :], self.visual[idx, :, :], self.text[idx, :, :],
                np.argmax(self.labels[idx], axis=-1)
            ]

        def __len__(self):
            return self.audio.shape[0]

    data_path = os.path.join(data_dir, "Archive/iemocap_data.pkl")
    iemocap_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    iemocap_train, iemocap_valid, iemocap_test = iemocap_data['train'], iemocap_data['valid'], iemocap_data['test']

    train_audio, train_visual, train_text, train_labels = iemocap_train[
                                                              AUDIO_], iemocap_train[VISUAL_], iemocap_train[TEXT_], \
                                                          iemocap_train[LABEL_]
    valid_audio, valid_visual, valid_text, valid_labels = iemocap_valid[
                                                              AUDIO_], iemocap_valid[VISUAL_], iemocap_valid[TEXT_], \
                                                          iemocap_valid[LABEL_]
    test_audio, test_visual, test_text, test_labels = iemocap_test[
                                                          AUDIO_], iemocap_test[VISUAL_], iemocap_test[TEXT_], \
                                                      iemocap_test[LABEL_]

    train_audio[train_audio == -np.inf] = 0
    valid_audio[valid_audio == -np.inf] = 0
    test_audio[test_audio == -np.inf] = 0

    ######################
    # Only MMT need this #
    ######################
    # transfer = MinMaxScaler(feature_range=(0, 1))
    # train_samples = train_text.shape[0]
    # valid_samples = valid_text.shape[0]
    # test_samples = test_text.shape[0]
    # lens = train_text.shape[1]
    # text_dim = train_text.shape[2]
    # visual_dim = train_visual.shape[2]
    # audio_dim = train_audio.shape[2]
    #
    # train_text = transfer.fit_transform(train_text.reshape(train_samples, lens * text_dim))
    # train_visual = transfer.fit_transform(train_visual.reshape(train_samples, lens * visual_dim))
    # train_audio = transfer.fit_transform(train_audio.reshape(train_samples, lens * audio_dim))
    #
    # valid_text = transfer.fit_transform(valid_text.reshape(valid_samples, lens * text_dim))
    # valid_visual = transfer.fit_transform(valid_visual.reshape(valid_samples, lens * visual_dim))
    # valid_audio = transfer.fit_transform(valid_audio.reshape(valid_samples, lens * audio_dim))
    #
    # test_text = transfer.fit_transform(test_text.reshape(test_samples, lens * text_dim))
    # test_visual = transfer.fit_transform(test_visual.reshape(test_samples, lens * visual_dim))
    # test_audio = transfer.fit_transform(test_audio.reshape(test_samples, lens * audio_dim))
    #
    # train_text = train_text.reshape(train_samples, lens, text_dim)
    # train_visual = train_visual.reshape(train_samples, lens, visual_dim)
    # train_audio = train_audio.reshape(train_samples, lens, audio_dim)
    #
    # valid_text = valid_text.reshape(valid_samples, lens, text_dim)
    # valid_visual = valid_visual.reshape(valid_samples, lens, visual_dim)
    # valid_audio = valid_audio.reshape(valid_samples, lens, audio_dim)
    #
    # test_text = test_text.reshape(test_samples, lens, text_dim)
    # test_visual = test_visual.reshape(test_samples, lens, visual_dim)
    # test_audio = test_audio.reshape(test_samples, lens, audio_dim)
    ######################

    print("train_audio.shape is  {}".format(train_audio.shape))
    print("train_visual.shape is {}".format(train_visual.shape))
    print("train_text.shape is   {}".format(train_text.shape))
    print("train_labels.shape is {}".format(train_labels.shape))
    print("valid_audio.shape is  {}".format(valid_audio.shape))
    print("valid_visual.shape is {}".format(valid_visual.shape))
    print("valid_text.shape is   {}".format(valid_text.shape))
    print("valid_labels.shape is {}".format(valid_labels.shape))
    print("test_audio.shape is  {}".format(test_audio.shape))
    print("test_visual.shape is {}".format(test_visual.shape))
    print("test_text.shape is   {}".format(test_text.shape))
    print("test_labels.shape is {}".format(test_labels.shape))

    # code that instantiates the Dataset objects
    train_set = IEMOCAP(train_audio, train_visual, train_text, train_labels)
    valid_set = IEMOCAP(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = IEMOCAP(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    # remove possible NaN values
    train_set.visual[train_set.visual != train_set.visual] = 0
    valid_set.visual[valid_set.visual != valid_set.visual] = 0
    test_set.visual[test_set.visual != test_set.visual] = 0

    train_set.audio[train_set.audio != train_set.audio] = 0
    valid_set.audio[valid_set.audio != valid_set.audio] = 0
    test_set.audio[test_set.audio != test_set.audio] = 0

    return train_set, valid_set, test_set, input_dims


def load_mosi(data_dir):
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels, meta):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels
            self.meta = meta

        def __getitem__(self, idx):
            return [
                self.audio[idx, :, :], self.visual[idx, :, :],
                self.text[idx, :, :], self.labels[idx],
                self.meta[idx][0].decode('UTF-8')
            ]

        def __len__(self):
            return self.audio.shape[0]

    data_path = os.path.join(data_dir, "Archive/mosi_data.pkl")
    mosi_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    mosi_train, mosi_valid, mosi_test = mosi_data['train'], mosi_data[
        'valid'], mosi_data['test']

    train_audio, train_visual, train_text, train_labels = mosi_train[
                                                              AUDIO_], mosi_train[VISUAL_], mosi_train[TEXT_], \
                                                          mosi_train[
                                                              LABEL_]
    valid_audio, valid_visual, valid_text, valid_labels = mosi_valid[
                                                              AUDIO_], mosi_valid[VISUAL_], mosi_valid[TEXT_], \
                                                          mosi_valid[
                                                              LABEL_]
    test_audio, test_visual, test_text, test_labels = mosi_test[
                                                          AUDIO_], mosi_test[VISUAL_], mosi_test[TEXT_], mosi_test[
                                                          LABEL_]

    train_meta = mosi_train['id']
    valid_meta = mosi_valid['id']
    test_meta = mosi_test['id']

    train_labels = np.squeeze(train_labels, axis=2)
    valid_labels = np.squeeze(valid_labels, axis=2)
    test_labels = np.squeeze(test_labels, axis=2)

    print("train_audio.shape is  {}".format(train_audio.shape))
    print("train_visual.shape is {}".format(train_visual.shape))
    print("train_text.shape is   {}".format(train_text.shape))
    print("train_labels.shape is {}".format(train_labels.shape))
    print("valid_audio.shape is  {}".format(valid_audio.shape))
    print("valid_visual.shape is {}".format(valid_visual.shape))
    print("valid_text.shape is   {}".format(valid_text.shape))
    print("valid_labels.shape is {}".format(valid_labels.shape))
    print("test_audio.shape is  {}".format(test_audio.shape))
    print("test_visual.shape is {}".format(test_visual.shape))
    print("test_text.shape is   {}".format(test_text.shape))
    print("test_labels.shape is {}".format(test_labels.shape))

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio, train_visual, train_text, train_labels, train_meta)
    valid_set = MOSI(valid_audio, valid_visual, valid_text, valid_labels, valid_meta)
    test_set = MOSI(test_audio, test_visual, test_text, test_labels, test_meta)

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    return train_set, valid_set, test_set, input_dims


def load_mosei(data_dir):
    class MOSEI(Dataset):
        '''
        PyTorch Dataset for MOSEI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [
                self.audio[idx, :, :], self.visual[idx, :, :],
                self.text[idx, :, :], self.labels[idx]
            ]

        def __len__(self):
            return self.audio.shape[0]

    data_path = os.path.join(data_dir, "Archive/mosei_senti_data.pkl")
    mosei_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    mosei_train, mosei_valid, mosei_test = mosei_data['train'], mosei_data[
        'valid'], mosei_data['test']

    train_audio, train_visual, train_text, train_labels = mosei_train[
                                                              AUDIO_], mosei_train[VISUAL_], mosei_train[TEXT_], \
                                                          mosei_train[
                                                              LABEL_]
    valid_audio, valid_visual, valid_text, valid_labels = mosei_valid[
                                                              AUDIO_], mosei_valid[VISUAL_], mosei_valid[TEXT_], \
                                                          mosei_valid[
                                                              LABEL_]
    test_audio, test_visual, test_text, test_labels = mosei_test[
                                                          AUDIO_], mosei_test[VISUAL_], mosei_test[TEXT_], mosei_test[
                                                          LABEL_]

    train_labels = np.squeeze(train_labels, axis=2)
    valid_labels = np.squeeze(valid_labels, axis=2)
    test_labels = np.squeeze(test_labels, axis=2)

    train_audio[train_audio == -np.inf] = 0
    valid_audio[valid_audio == -np.inf] = 0
    test_audio[test_audio == -np.inf] = 0

    ######################
    # Only MMT need this #
    ######################
    # transfer = MinMaxScaler(feature_range=(0, 1))
    # train_samples = train_text.shape[0]
    # valid_samples = valid_text.shape[0]
    # test_samples = test_text.shape[0]
    # lens = train_text.shape[1]
    # text_dim = train_text.shape[2]
    # visual_dim = train_visual.shape[2]
    # audio_dim = train_audio.shape[2]
    #
    # train_text = transfer.fit_transform(train_text.reshape(train_samples, lens * text_dim))
    # train_visual = transfer.fit_transform(train_visual.reshape(train_samples, lens * visual_dim))
    # train_audio = transfer.fit_transform(train_audio.reshape(train_samples, lens * audio_dim))
    #
    # valid_text = transfer.fit_transform(valid_text.reshape(valid_samples, lens * text_dim))
    # valid_visual = transfer.fit_transform(valid_visual.reshape(valid_samples, lens * visual_dim))
    # valid_audio = transfer.fit_transform(valid_audio.reshape(valid_samples, lens * audio_dim))
    #
    # test_text = transfer.fit_transform(test_text.reshape(test_samples, lens * text_dim))
    # test_visual = transfer.fit_transform(test_visual.reshape(test_samples, lens * visual_dim))
    # test_audio = transfer.fit_transform(test_audio.reshape(test_samples, lens * audio_dim))
    #
    # train_text = train_text.reshape(train_samples, lens, text_dim)
    # train_visual = train_visual.reshape(train_samples, lens, visual_dim)
    # train_audio = train_audio.reshape(train_samples, lens, audio_dim)
    #
    # valid_text = valid_text.reshape(valid_samples, lens, text_dim)
    # valid_visual = valid_visual.reshape(valid_samples, lens, visual_dim)
    # valid_audio = valid_audio.reshape(valid_samples, lens, audio_dim)
    #
    # test_text = test_text.reshape(test_samples, lens, text_dim)
    # test_visual = test_visual.reshape(test_samples, lens, visual_dim)
    # test_audio = test_audio.reshape(test_samples, lens, audio_dim)
    ######################

    print("train_audio.shape is  {}".format(train_audio.shape))
    print("train_visual.shape is {}".format(train_visual.shape))
    print("train_text.shape is   {}".format(train_text.shape))
    print("train_labels.shape is {}".format(train_labels.shape))
    print("valid_audio.shape is  {}".format(valid_audio.shape))
    print("valid_visual.shape is {}".format(valid_visual.shape))
    print("valid_text.shape is   {}".format(valid_text.shape))
    print("valid_labels.shape is {}".format(valid_labels.shape))
    print("test_audio.shape is  {}".format(test_audio.shape))
    print("test_visual.shape is {}".format(test_visual.shape))
    print("test_text.shape is   {}".format(test_text.shape))
    print("test_labels.shape is {}".format(test_labels.shape))

    # code that instantiates the Dataset objects
    train_set = MOSEI(train_audio, train_visual, train_text, train_labels)
    valid_set = MOSEI(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = MOSEI(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    text_dim = train_set[0][2].shape[1]
    print("Text feature dimension is: {}".format(text_dim))
    input_dims = (audio_dim, visual_dim, text_dim)

    return train_set, valid_set, test_set, input_dims


def load_mosi_bert(data_dir):
    class MOSI(Dataset):
        '''
        PyTorch Dataset for MOSI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [
                torch.Tensor(self.audio[idx]), torch.Tensor(self.visual[idx]),
                torch.Tensor(self.text[idx]), torch.Tensor(self.labels[idx])
            ]

        def __len__(self):
            return len(self.labels)

    data_path = os.path.join(data_dir, 'MOSI/aligned_50.pkl')
    mosi_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    mosi_train, mosi_valid, mosi_test = mosi_data['train'], mosi_data[
        'valid'], mosi_data['test']

    train_audio, train_visual, train_text, train_labels = mosi_train[
                                                              AUDIO_].astype(np.float32), mosi_train[VISUAL_].astype(
        np.float32), mosi_train[BERT].astype(np.float32), \
                                                          mosi_train[
                                                              BERT_LABEL][:, np.newaxis].astype(np.float32)
    valid_audio, valid_visual, valid_text, valid_labels = mosi_valid[
                                                              AUDIO_].astype(np.float32), mosi_valid[VISUAL_].astype(
        np.float32), mosi_valid[BERT].astype(np.float32), \
                                                          mosi_valid[
                                                              BERT_LABEL][:, np.newaxis].astype(np.float32)
    test_audio, test_visual, test_text, test_labels = mosi_test[
                                                          AUDIO_].astype(np.float32), mosi_test[VISUAL_].astype(
        np.float32), mosi_test[BERT].astype(np.float32), mosi_test[
                                                             BERT_LABEL][:, np.newaxis].astype(np.float32)

    train_audio[train_audio == -np.inf] = 0
    valid_audio[valid_audio == -np.inf] = 0
    test_audio[test_audio == -np.inf] = 0

    print("train_audio.shape is  {}".format(train_audio.shape))
    print("train_visual.shape is {}".format(train_visual.shape))
    print("train_text.shape is   {}".format(train_text.shape))
    print("train_labels.shape is {}".format(train_labels.shape))
    print("valid_audio.shape is  {}".format(valid_audio.shape))
    print("valid_visual.shape is {}".format(valid_visual.shape))
    print("valid_text.shape is   {}".format(valid_text.shape))
    print("valid_labels.shape is {}".format(valid_labels.shape))
    print("test_audio.shape is  {}".format(test_audio.shape))
    print("test_visual.shape is {}".format(test_visual.shape))
    print("test_text.shape is   {}".format(test_text.shape))
    print("test_labels.shape is {}".format(test_labels.shape))

    # code that instantiates the Dataset objects
    train_set = MOSI(train_audio, train_visual, train_text, train_labels)
    valid_set = MOSI(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = MOSI(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    print("Text feature dimension is: {}".format(768))
    input_dims = (audio_dim, visual_dim, 768)

    return train_set, valid_set, test_set, input_dims


def load_mosei_bert(data_dir):
    class MOSEI(Dataset):
        '''
        PyTorch Dataset for MOSEI, don't need to change this
        '''

        def __init__(self, audio, visual, text, labels):
            self.audio = audio
            self.visual = visual
            self.text = text
            self.labels = labels

        def __getitem__(self, idx):
            return [
                torch.Tensor(self.audio[idx]), torch.Tensor(self.visual[idx]),
                torch.Tensor(self.text[idx]), torch.Tensor(self.labels[idx])
            ]

        def __len__(self):
            return self.audio.shape[0]

    data_path = os.path.join(data_dir, 'MOSEI/aligned_50.pkl')
    mosei_data = pickle.load(open(data_path, 'rb'), encoding='latin1')
    mosei_train, mosei_valid, mosei_test = mosei_data['train'], mosei_data[
        'valid'], mosei_data['test']

    train_audio, train_visual, train_text, train_labels = mosei_train[
                                                              AUDIO_].astype(np.float32), mosei_train[VISUAL_].astype(
        np.float32), mosei_train[BERT].astype(np.float32), \
                                                          mosei_train[
                                                              BERT_LABEL][:, np.newaxis].astype(np.float32)
    valid_audio, valid_visual, valid_text, valid_labels = mosei_valid[
                                                              AUDIO_].astype(np.float32), mosei_valid[VISUAL_].astype(
        np.float32), mosei_valid[BERT].astype(np.float32), \
                                                          mosei_valid[
                                                              BERT_LABEL][:, np.newaxis].astype(np.float32)
    test_audio, test_visual, test_text, test_labels = mosei_test[
                                                          AUDIO_].astype(np.float32), mosei_test[VISUAL_].astype(
        np.float32), mosei_test[BERT].astype(np.float32), mosei_test[
                                                              BERT_LABEL][:, np.newaxis].astype(np.float32)

    train_audio[train_audio == -np.inf] = 0
    valid_audio[valid_audio == -np.inf] = 0
    test_audio[test_audio == -np.inf] = 0

    ######################
    # Only MMT need this #
    ######################
    # transfer = MinMaxScaler(feature_range=(0, 1))
    # train_samples = train_text.shape[0]
    # valid_samples = valid_text.shape[0]
    # test_samples = test_text.shape[0]
    # lens = train_text.shape[2]
    # visual_dim = train_visual.shape[2]
    # audio_dim = train_audio.shape[2]
    #
    # train_visual = transfer.fit_transform(train_visual.reshape(train_samples, lens * visual_dim))
    # train_audio = transfer.fit_transform(train_audio.reshape(train_samples, lens * audio_dim))
    #
    # valid_visual = transfer.fit_transform(valid_visual.reshape(valid_samples, lens * visual_dim))
    # valid_audio = transfer.fit_transform(valid_audio.reshape(valid_samples, lens * audio_dim))
    #
    # test_visual = transfer.fit_transform(test_visual.reshape(test_samples, lens * visual_dim))
    # test_audio = transfer.fit_transform(test_audio.reshape(test_samples, lens * audio_dim))
    #
    # train_visual = train_visual.reshape(train_samples, lens, visual_dim)
    # train_audio = train_audio.reshape(train_samples, lens, audio_dim)
    #
    # valid_visual = valid_visual.reshape(valid_samples, lens, visual_dim)
    # valid_audio = valid_audio.reshape(valid_samples, lens, audio_dim)
    #
    # test_visual = test_visual.reshape(test_samples, lens, visual_dim)
    # test_audio = test_audio.reshape(test_samples, lens, audio_dim)
    ######################

    print("train_audio.shape is  {}".format(train_audio.shape))
    print("train_visual.shape is {}".format(train_visual.shape))
    print("train_text.shape is   {}".format(train_text.shape))
    print("train_labels.shape is {}".format(train_labels.shape))
    print("valid_audio.shape is  {}".format(valid_audio.shape))
    print("valid_visual.shape is {}".format(valid_visual.shape))
    print("valid_text.shape is   {}".format(valid_text.shape))
    print("valid_labels.shape is {}".format(valid_labels.shape))
    print("test_audio.shape is  {}".format(test_audio.shape))
    print("test_visual.shape is {}".format(test_visual.shape))
    print("test_text.shape is   {}".format(test_text.shape))
    print("test_labels.shape is {}".format(test_labels.shape))

    # code that instantiates the Dataset objects
    train_set = MOSEI(train_audio, train_visual, train_text, train_labels)
    valid_set = MOSEI(valid_audio, valid_visual, valid_text, valid_labels)
    test_set = MOSEI(test_audio, test_visual, test_text, test_labels)

    audio_dim = train_set[0][0].shape[1]
    print("Audio feature dimension is: {}".format(audio_dim))
    visual_dim = train_set[0][1].shape[1]
    print("Visual feature dimension is: {}".format(visual_dim))
    print("Text feature dimension is: {}".format(768))
    input_dims = (audio_dim, visual_dim, 768)

    return train_set, valid_set, test_set, input_dims
