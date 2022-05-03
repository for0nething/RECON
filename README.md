# RECON



# Quick Start

## Folder Structure

    .
    ├── RECON                   # RECON codes for coreset construction
    ├── MLModel                 # ML models training codes to test the performance of RECON
    └── README.md               



## Requirements
Before running the codes, please make sure your C++ version is above `C++14`. 
Library cnpy is also needed to save results in the format of npz.

The dataset path is configured by variable `DATAPATH` (line 13 in  global.h), which should also be configured properly before running the codes.
The datasets can be downloaded from [dataset link](https://cloud.tsinghua.edu.cn/d/96132c6b279e4097baaa/).
- `Python 3.7+`

- ` C++14`
- `cnpy: a library to read/write .npy and .npz files in C/C++`  [link](https://github.com/rogersce/cnpy)



## Usage

### RECON on IMDB / IMDB-Large:
Build `./RECON` first, and then perform RECON on different datasets by passing different arguments.
> parameter setting:  
>> [dataName] [proportion] [0:IMDB 1:IMDB-Large] [0:Classification 1:Regression]

- `IMDB,  p=0.0128 for classification:   RECON IMDB 0.0128 0 0 `
- `IMDB,  p=0.0032 for regression:   RECON IMDB 0.0032 0 1`
- `IMDB-Large, p=0.0016 for classification: RECON IMDB 0.0016 1 0`
- `IMDB-Large, p=0.0016 for regression RECON IMDB: 0.0016 1 1`



### RECON on stack / Brazil / taxi:


> parameter setting:  
>> [dataName] [proportion] 
- `stack, p=0.0032: RECON stack 0.0032`
- `Brazil, p=0.0016: RECON Brazil 0.0016`
- `taxi, p=0.0032: RECON taxi 0.0032`

>  Note: '-L/usr/local/lib/ -lcnpy -lz' may also need to add to the program arguments, depending on the method to install cnpy.

**Note:** Before running RECON, make sure the variable `DATAPATH` (line 13 in  global.h) is configured as the path of dataset.
Besides, make sure the vaiable `CSPATH` (line 14 in gloabl.h) is configured as the location to save RECON's output, i.e., coresets.


### Training Logistic Regression
Run `./MLModel/logsitic-universal.py` to train logistic regression models.

- IMDB: `python logistic-universal.py --data IMDBC5 --method sgd -s 0.0128 `

- IMDB-Large: `python logistic-universal.py --data IMDBLargeC5 --method sgd -s 0.0016 `


- Brazil: `python logistic-universal.py --data Brazilnew --method sgd -s 0.0016 `

 

### Training Linear Regression
Run `./MLModel/linear-universal.py` to train linear regression models.

- IMDB: `python linear-universal.py --data IMDBCLinear --method sgd -s 0.0032 `

- IMDB-Large: `python linear-universal.py --data IMDBLargeCLinear --method sgd -s 0.0016 `

- stack: `python linear-universal.py --data stackn --method sgd -s 0.0032`


- taxi: `python linear-universal.py --data taxi --method sgd -s 0.0032`

**Note:** Before training models, make sure variable `DATAPATH` (line 1 in  Global.py) is configured as the path of datasets. 
And `CSPATH`(line 2 in  Global.py) is configured as the path to RECON's output (path of coreset).  


## License

The project is available under the [MIT](LICENSE) license.

## Contact
If you have any questions, feel free contact us.
