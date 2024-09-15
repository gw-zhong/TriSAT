![Python 3.8](https://img.shields.io/badge/python-3.8-green)

>Codes for **[TriSAT: Trimodal Representation Learning for Multimodal Sentiment Analysis](https://ieeexplore.ieee.org/document/10675444)** （Accepted by IEEE/ACM Transactions on Audio, Speech and Language Processing）.

## Usage
### Clone the repository
    git clone https://github.com/gw-zhong/TriSAT.git
### Download the datasets and BERT models
+ [CMU-MOSI & CMU-MOSEI (**Glove**) [align & unaligned]](http://immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/) (which are not available now)

+ [CMU-MOSI & CMU-MOSEI (**BERT**) [align & unaligned]](https://github.com/thuiar/MMSA)

Alternatively, you can download these datasets from:
- [BaiduYun Disk](https://pan.baidu.com/s/16UcDXgwmq9kxHf6ziJcChw) ```code: zpqk```

For convenience, we also provide the BERT pre-training model that we fine-tuned with:

- [BaiduYun Disk](https://pan.baidu.com/s/12zhRpTEx5589Bmo0OAF5cg) ```code: e7mw```

### Preparation
Create (empty) folders for data, results, and models:
 ```python
cd TriSAT
 mkdir input results models
```
and put the downloaded data in 'input/'.

### Run the code
 ```python
python main_[DatasetName].py [--FLAGS]
 ```

## Contact
If you have any question, feel free to contact me through [guoweizhong@zjut.edu.cn](guoweizhong@zjut.edu.cn).