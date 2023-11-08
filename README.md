# COCOA: framework to map fine-scale cell-type-specific chromatin compartmentalization with epigenomic information


## Summary

COCOA is a a deep neural network framework based on convolution and attention, to predict  reliable fine-scale chromatin compartment patterns from six representative histone modification signals. COCOA can accurately infer chromatin compartmentalization at fine-scale resolution and present stable performance on test sets. In addition, we provide the PyTorch implementations for both training and predicting procedures.

### **_Note:_** To explore the detailed architecture of the COCOA please read the file _model.py_.


## Dependency

COCOA is written in Python3 with PyTorch framework.

The following versions are recommended when using iEnhance:

- torch 1.12.1
- numpy 1.21.2
- scipy 1.7.1
- pandas 1.3.3
- scikit-learn 1.0.2
- matplotlib 3.1.0
- tqdm 4.62.3
- cooler 0.8.11

**_Note:_** GPU usage for training and testing is highly recommended.


## Data Preparation

### 1. ChIP-seq data

For ChIP-seq data, we desire an input in _.bigWig_ format (*signal p-value*). If your data is other format, please use the format conversion tool to convert to  _.bigWig_ file format (*signal p-value*). All ChIP-seq data used in the paper are obtained from the ENCODE database.

When all the data were ready, they were stored in separate folders according to the different cell lines.
~~~shell
--HFFc6
 -H3K27ac.bigWig
 -H3K27me3.bigWig
 ...

--GM12878
 -H3K27ac.bigWig
 ...

...
~~~

Then configure the basic information of the script **data_pre.py**.

The following code blocks are the variables to be configured, **inp_dir** indicates the path where the stored ChIP-seq data is located, **in_resolution** indicates the predicion resolution of the chromatin compartment, **chrs_list** is the chromosome number to be calculated, and **out_dir** indicates the identifier of the outputs.
~~~python
chrs_list = ['12' ,'13' ,'14' ,'15' ,'16' ,'17' ,'18' ,'19']
in_resolution = 25000
inp_dir = "---" # example as : ../data/epi/
out_dir = "---" # example as : ../data/hff1/
~~~

When all preparations have been completed, execute the following command:
~~~bash
python data_pre.py
~~~

### 2. Micro-C/Hi-C data (This step is only necessary if you want to retrain COCOA.)

For Hi-C/Micro-C data, we desire an input in _.cool_ file format. If your data is in _.hic_ format, please use the format conversion tool to convert to _.cool_ file format.

## Usage

## **_Note:_** Due to historical legacy issues, the **_from MFGenhffh import MFGen_** code statement in all training and prediction code is forbidden to be removed, otherwise it will cause the script to crash!.

### 1. Predicting
To execute the prediction script of COCOA, first configure the basic information of the script **predict.py**.

The following code blocks are the variables to be configured, **model_path** indicates the path where the pre-trained model is located, **inp_dir** indicates the path of the inputs (Output results processed in the *ChIP-seq data preparation* step), **chrs_list** is the chromosome number to be calculated, and **save_path** indicates the identifier of the final output result.
~~~python
model_path = "---" # example as : ./pretrained/RegModule-Best.pt
inp_dir = "---" # example as : ../data/hff1/
chrs_list = ["18"] 
save_path = "./" 
~~~

When all preparations have been completed, execute the following command:
~~~bash
python predict.py
~~~

### 2. Applying predicted data
The output predictions are stored in *.npz* files that store numpy arrays under keys.
To apply the generated compartment patterns, use the following command in a python file: 
~~~python
cm_matrix = np.load("path/to/file.npz")['pre_out']
~~~

### 3. Training
To retrain COCOA, you need to build a training data first.

1. Pairing the ChIP-seq data with Micro-C (Hi-C) data (these data have obtained in the *Data Preparation* step) and separately storing each chromsome in *.npz* format.

The following code blocks are the variables to be configured, **micro_path** indicates the path of the Micro-C or Hi-C data; **epi_path** indicates the path of the ChIP-seq data; **chrs_list** is the chromosome number to be calculated,; **store_dir** indicates the name of the folder where the output file is stored.
~~~python
micro_path = "/path/to/micro-c/data"
epi_path = "---" # example as : ../data/epi/hff1/
chrs_list = ['5' ,'6' ,'7' ,'8' ,'9' ]
store_dir = "---"
~~~
>**_Note:_** Micro-C data and ChIP-seq data resolution should be consistent.

Finally, execute the following command:
~~~bash
python divide.py
~~~

2. Constructing train and test sets.

Specify the datasets dividing and give the storage path of the previous step in the `script`, then execute the following command:
~~~bash
python construct.py
~~~

3. Training

When you have completed all the above steps, you can retrain the model with the following command:
~~~bash
python train.py 0
~~~
> Note: If your training is interrupted or you want to continue training from an epoch, change the script parameters and execute `python train.py 1`.
