## 人脸3D建模算法

 



写在前面：

```
系统：windows
内存：16G
显卡：nvidia geforce gtx 1650
cpu: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 
虚拟环境配置：anaconda
使用设备：cpu(如果想使用gpu加速，可以根据网上教程，根据自己的电脑显卡驱动版本和搭建的环境中torch的版本下载对应版本的cuda)
预计代码运行时间：30分钟
```



### 一、虚拟环境搭建与资源下载

​	1. 下载anaconda

​	2. 在anaconda创建新的虚拟环境

```
numpy              1.23.0
scipy              1.4.1
chumpy             0.70
scikit-image       0.15.0
opencv-python      4.1.2.30
PyYAML             5.1.1
torch              1.8.2
torchvision        0.9.2
face-alignment     1.3.5
yacs               0.1.8
kornia             0.4.1
ninja              1.11.1
fvcore             0.1.5.post20220512
pytorch3d          0.7.0
```

注：torch安装和pytorch3d的安装请找好教程安装，版本或者是安装步骤容易出错，尽量使用anaconda的虚拟环境，尽量不要在本机安装。)，否则会出现版本不兼容的情况



### 二、代码的配置与执行

 	1. 第一步，在虚拟环境中运行如下指令参数

```
python demos\demo_reconstruct.py --inputpath TestSamples\examples --savefolder TestSamples\examples\results 
--saveDepth True --saveObj True --device cpu --rasterizer_type pytorch3d --useTex True
```

如果使用pycharm等ide，则在demo_reconstruct.py的configure中配置

```
--inputpath ..\TestSamples\examples
--savefolder ..\TestSamples\examples\results
--saveDepth True
--saveObj True
--device cpu
--rasterizer_type pytorch3d
--useTex True
```

参数说明：

```
inputpath：输入的图片路径
savefolder：保存结果路径
其他配置在代码中有说明，不在赘述
```

注：若ide出现找不到指定模块的错误例如：

```
ImportError: DLL load failed while importing win32file: 找不到指定的模块
```

则在Run Configuration的Environment Variables设置

```
CONDA_DLL_SEARCH_MODIFICATION_ENABLE = 1
```



2. 第二步，在虚拟环境中运行如下指令参数

```
python demos\demo_transfer.py --image_path TestSamples\examples\xxx.jpg --exp_path TestSamples\exp\x.jpg
--savefolder TestSamples\examples\results --device cpu --rasterizer_type pytorch3d
```

如果使用pycharm等ide，则在demo_transfer.py的configure中配置

```
--image_path ..\TestSamples\examples\xxx.jpg
--exp_path ..\TestSamples\exp\x.jpg
--savefolder ..\TestSamples\examples\results
--device cpu
--rasterizer_type pytorch3d
```

参数说明：

```
image_path：输入图片地址，xxx改为路径中存在的图片名称
exp_path：迁移的表情图片，x为选择表情图片的序号
savefolder：保存结果路径
其他参数不在赘述
```

注：若ide出现找不到指定模块的错误例如：

```
ImportError: DLL load failed while importing win32file: 找不到指定的模块
```

则在Run Configuration的Environment Variables设置

```
CONDA_DLL_SEARCH_MODIFICATION_ENABLE = 1
```



3. 第三步，在虚拟环境中运行如下指令参数

```
python demos\demo_teaser.py --inputpath TestSamples\examples\xxx.jpg --exp_path TestSamples\exp
--savefolder TestSamples\teaser\results --device cpu --rasterizer_type pytorch3d
```

如果使用pycharm等ide，则在demo_teaser.py的configure中配置

```
--inputpath ..\TestSamples\examples\xxx.jpg
--exp_path ..\TestSamples\exp
--savefolder ..\TestSamples\teaser\results
--device cpu
--rasterizer_type pytorch3d
```

参数说明：

```
image_path：输入图片地址，xxx改为路径中存在的图片名称
exp_path：迁移的表情图片的路径
savefolder：保存结果路径
其他参数不在赘述
```

注：若ide出现找不到指定模块的错误例如：

```
ImportError: DLL load failed while importing win32file: 找不到指定的模块
```

则在Run Configuration的Environment Variables设置

```
CONDA_DLL_SEARCH_MODIFICATION_ENABLE = 1
```



### 三、代码执行中的输入与输出

1. 运行demo_reconstruct.py

   运行参数

   ```
   python demos\demo_reconstruct.py --inputpath TestSamples\examples --savefolder TestSamples\examples\results 
   --saveDepth True --saveObj True --device cpu --rasterizer_type pytorch3d --useTex True
   ```

   输入：inputpath配置的路径中的图片，对图片循环进行粗糙人脸重建。

   输出：人脸重建的结果或放入savefolder中，同时会吧代码中设置要保存的其他类型的文件一并导出

   注：保存的其他文件，例如.obj文件会一并输出，注意查看输出路径中的结果。

   2. 运行demo_transfer.py

   运行参数

   ```
python demos\demo_transfer.py --image_path TestSamples\examples\xxx.jpg --exp_path TestSamples\exp\x.jpg
   --savefolder TestSamples\examples\results --device cpu --rasterizer_type pytorch3d
   ```
```
   
输入：inputpath配置的路径中的选定的图片，对图片进行精细的人脸重建，exp_path为表情迁移的图片，将表情迁移到inputpath的图片上
   

​	输出：人脸重建的结果或放入savefolder中，同时会吧代码中设置要保存的其他类型的文件一并导出

注：保存的其他文件，例如.obj文件会一并输出，注意查看输出路径中的结果。

3. 运行demo_teaser.py

   运行参数

```
   python demos\demo_teaser.py --inputpath TestSamples\examples\xxx.jpg --exp_path TestSamples\exp
   --savefolder TestSamples\teaser\results --device cpu --rasterizer_type pytorch3d
   ```

   输入：inputpath配置的路径中的选定的图片，对图片进行精细的人脸重建，exp_path为表情迁移的图片，将表情迁移到inputpath的图片上

   输出：


注：保存的其他文件，例如.obj文件会一并输出，注意查看输出路径中的结果。


   ```