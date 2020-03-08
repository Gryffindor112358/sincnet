# Sincnet终极说明
## 1、数据预处理TIMIT_preparation.py
&emsp;&emsp;这部分是把TIMIT数据集进行一些预处理，主要是把每个音频信号中的空白音去除掉，然后做一些normalization什么的，把处理后的数据放在文件夹OUTPUT_FOLDER中
&emsp;&emsp;这个文件的执行命令是
&emsp;&emsp;python TIMIT_preparation.py TIMIT_FOLDER OUTPUT_FOLDER data_lists/TIMIT_all.scp
&emsp;&emsp;其中TIMIT_FOLDER是原始数据集的文件夹，OUTPUT_FOLDER不用创建，代码会自己创建
## 2、train.py
&emsp;&emsp;这个是最终训练出来模型的文件，就是一般的流程，创建模型，给模型compile一些参数，最关键的是代码最后面，train_generator把本模型所需的数据都输入进去，最后model.fit训练

&emsp;&emsp;执行命令是python train.py --cfg=cfg/SincNet_TIMIT.cfg（这个SincNet_TIMTI.cfg在4中说明
##3、test.py
&emsp;&emsp;这里面主要是需要看懂预测部分的逻辑与操作。
&emsp;&emsp;执行主要功能的是validate函数，在此函数里对每一个信号进行循环处理，先根据窗的长度和移动窗的长度将一个信号分成N_fr段。具体执行的时候往sig_arr中装填分块后的信号，每装满Batch_dev块之后进行一次预测，预测结果的放在pout中，这个数据的宽度是标签的个数，即对每一段信号的预测结果都是一个数组，数据为对应每一个标签的概率值
&emsp;&emsp;此番操作之后每一个信号都会得到一个pout，行数为信号分段数，即每一段信号都有一个预测结果，最终选择一个最优的作为此信号的预测结果。
test.py的功能呗train.py调用了，我们掌握这些主要是为了手动写预测部分的代码
##4、读取数据事宜
&emsp;&emsp;这个工程中的数据读取采用较为集成统一的方式，在执行train.py时调用的配置文件SincNet_TIMIT.cfg中写满了此工程中所需的所有数据读取的地址一个模型中需要使用的参数。
&emsp;&emsp;在data_io.py中定义了函数read_conf中读取了上述地址的文件中的数据以及各种参数

&emsp;&emsp;在conf.py文件中把这些所有的数据都给赋予了相应的变量，供其他文件调用
