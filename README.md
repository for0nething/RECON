# RECON
This repo contains the codes for the VLDB 2023 paper [_Coresets over multiple tables for feature-rich and data-efficient machine learning_](https://www.vldb.org/pvldb/vol16/p64-wang.pdf). 


# Quick Start

## Folder Structure

    .
    ├── preprocess              # Data preprocessing codes
    ├── RECON                   # RECON codes for coreset construction
    ├── MLModel                 # ML models training codes to test the performance of RECON
    ├── linear-universal.py     # Evaluation of regression models
    ├── logistic-universal.py   # Evaluation of classification models
    └── README.md               



## Requirements
Before running the codes, please make sure your C++ version is above `C++14`. 
Library cnpy is also needed to save results in the format of npz.

The dataset path is configured by variable `DATAPATH` (line 9 in  global.h), which should also be configured properly before running the codes.
The datasets can be downloaded from [dataset link](https://drive.google.com/drive/folders/1kOLJQRnJk-_87y3WVq8Dwu18JYylbQhb?usp=sharing).
- `Python 3.7+`

- ` C++14`
- `cnpy: a library to read/write .npy and .npz files in C/C++`  [link](https://github.com/rogersce/cnpy)



## Usage

### RECON on IMDB / IMDB-Large:
First build `./RECON` by:

- `cd RECON`

- `cmake .`

- `make`


and then perform RECON on different datasets by passing different arguments.
> parameter setting:  
>> [dataName] [proportion] [0:IMDB 1:IMDB-Large] [0:Classification 1:Regression]

- `IMDB,  p=0.0128 for classification:   ./RECON IMDB 0.0128 0 0 `
- `IMDB,  p=0.0032 for regression:   ./RECON IMDB 0.0032 0 1`
- `IMDB-Large, p=0.0016 for classification: ./RECON IMDB 0.0016 1 0`
- `IMDB-Large, p=0.0016 for regression ./RECON IMDB 0.0016 1 1`



### RECON on stack / Brazil / taxi:


> parameter setting:  
>> [dataName] [proportion] 
- `stack, p=0.0032: ./RECON stack 0.0032`
- `Brazil, p=0.0016: ./RECON Brazil 0.0016`
- `taxi, p=0.0032: ./RECON taxi 0.0032`

>  Note: '-L/usr/local/lib/ -lcnpy -lz' may also need to be added to the program arguments, which depends on the method to install cnpy.

**Note:** Before running RECON, make sure the variable `DATAPATH` (line 9 in  global.h) is configured as the path of dataset.
Besides, make sure the vaiable `CSPATH` (line 10 in gloabl.h) is configured as the location to save RECON's output, i.e., coresets.


### Training Logistic Regression
Run `logsitic-universal.py` to train logistic regression models.

- IMDB: `python logistic-universal.py --data IMDBC5 --method sgd -s 0.0128 `

- IMDB-Large: `python logistic-universal.py --data IMDBLargeC5 --method sgd -s 0.0016 `


- Brazil: `python logistic-universal.py --data Brazilnew --method sgd -s 0.0016 `

 

### Training Linear Regression
Run `linear-universal.py` to train linear regression models.

- IMDB: `python linear-universal.py --data IMDBCLinear --method sgd -s 0.0032 `

- IMDB-Large: `python linear-universal.py --data IMDBLargeCLinear --method sgd -s 0.0016 `

- stack: `python linear-universal.py --data stackn --method sgd -s 0.0032`


- taxi: `python linear-universal.py --data taxi --method sgd -s 0.0032`

**Note:** Before training models, make sure variable `DATAPATH` (line 1 in  Global.py) is configured as the path of datasets. 
And `CSPATH`(line 2 in  Global.py) is configured as the path to RECON's output (path of coreset).  

### Other Baselines

- **Sample-Join**: The argument `--greedy [0:Uniform Sampling 1:Coreset (default)]` specifies the subset for training. 
Sample-Join can be achieved by setting `--greedy 0`.
For example, to train a logistic regression model on a uniform sampling of IMDB, you may use:
```sh
python logistic-universal.py --data IMDBC5 --method sgd -s 0.0128 --greedy 0
```


- **Full**: Full can be achieved by setting `-s` to `1` on top of Sample-Join. 
For example, to train a logistic regression model using full data of IMDB, you may use:
```sh
python logistic-universal.py --data IMDBC5 --method sgd -s 1 --greedy 0
```

- **Coreset-Join** and **Join-Coreset**: You can find their official implementations from [link](https://github.com/baharanm/craig).

### Data Preprocessing
In general, our preprocessing of each dataset in the `preprocess` directory can be summarized as data cleaning, normalization, and partition by label. 
We provide the preprocessed data in [dataset link](https://drive.google.com/drive/folders/1kOLJQRnJk-_87y3WVq8Dwu18JYylbQhb?usp=sharing).
The raw datasets can be found in their original sources.

## License

The project is available under the [MIT](LICENSE.md) license.

## Citation
If our work is helpful to you, please cite our [paper](https://www.vldb.org/pvldb/vol16/p64-wang.pdf):
```bibtex
@article{wang2022coresets,
  title={Coresets over multiple tables for feature-rich and data-efficient machine learning},
  author={Wang, Jiayi and Chai, Chengliang and Tang, Nan and Liu, Jiabin and Li, Guoliang},
  journal={Proceedings of the VLDB Endowment},
  volume={16},
  number={1},
  pages={64--76},
  year={2022},
  publisher={VLDB Endowment}
}

```
