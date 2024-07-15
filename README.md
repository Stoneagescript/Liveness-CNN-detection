# Liveness-CNN-detection
Eng:A pre trained open-source liveness detection model used as a prerequisite for any face detection or recognition system. CHN：一个训练好的开源活体检测模型，用于任意人脸检测或识别系统的前置条件，代码和模型仅为调试中产品  
The code and model are only debugging products, and you can adjust the code framework according to your own needs. I searched online and found that there are almost no free open-source liveness detection models available in China. So, I searched for information and created one myself. Of course, this is a pre condition for face recognition, so the focus of the code is not here. I will update some tutorials for full stack applications in the future.  
大家可以根据自己的需求调整代码框架，我搜了一下网上几乎没有，哦中国的网上几乎没有免费的活体检测开源模型，就自己查询资料制作了一款，当然我这里是用于人脸识别的前置条件，所以代码重心并不在此，后续会更新一些应用到全栈的教程。  

  压缩包里有所有文件夹的详细说明和指令调用过程，当然如何使用指令在脚本中也有注释，因为只用到opencv和tenserflow的关系，这边环境配置就不过多赘述，如果你准备做这个事情，那理应拥有所需要的配置，但还是要提一嘴，TensorFlow一定要2.0以上版本哦，否则训练时会出错。  
  还是想骂CSDN，啥都没有，啥都要收费，真别用了。  

  reflection.py 是只基于皮肤纹理对活体的检测，光线充足的环境（办公室等）精准度已足够，如果你部署的设备真的烂到要用这个脚本我只能祝你好运。  
  from skimage.feature import graycomatrix, graycoprops这里，如果你的版本太老，不妨试试from skimage.feature import greycomatrix, greycoprops，别忘了后续更改  
  也许会报关于opencv的错误，不过多半是让你别装headless版本，这个卸载再装就行。
