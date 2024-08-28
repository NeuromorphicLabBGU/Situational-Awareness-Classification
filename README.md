# Situational Awareness Classification Based on EEG signals and SNN

![](./res/net_arch.png)

**Autohrs:** Yakir Hadad, Moshe Bensimon, Shlomo Greenberg and Yehuda Ben-Shimol 

### Requirements
```bash
pip install -r requirements
```

## Dataset
The [Dataset](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection) is a collection of 34 experiments for monitoring of attention state in human individuals using passive EEG BCI.

Each Matlab file contains the object of the data acquired from EMOTIV device during one experiment. The raw data is contained in o.data, which is array of size {number-of-samples}x25, thus o.data(:,i) comprises one data channel. The sampling frequency is 128 Hz. The list of data channels and their numerical ids is given below per EMOTIV documentation;
The EEG data is in the channels 4:17.



### Usage
1. Download the dataset from [Dataset](https://www.kaggle.com/datasets/inancigdem/eeg-data-for-mental-attention-state-detection), into datasets folder.
2. Run `signal2spikes.py` to create a "new" encoded dataset of all the signals after feature extraction with the resonators.
3. run `train.py` to train the network with the encoded data.
