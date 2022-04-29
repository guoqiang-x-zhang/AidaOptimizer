# Aida Optimizer
## 1) Brief description of Aida
Aida optimiser is designed to effectively train DNN models. As an extension of AdaBelief, Aida exploits the layerwise gradient statistics via simple layerwise vector projections per iteration.  One property of Aida is that the histogram of the adaptive stepsizes tends to much more compact (or narrow) than those of AdaBelief and Adam when training a DNN model. See __G. Zhang, K. Niwa, and W. B. Kleijn, "On Exploiting Layerwise Gradient Statistics for Effective Training of Deep Neural Networks", arXiv:2203.13273v3, March, 2022.__ [[paper link]](https://arxiv.org/abs/2203.13273) 

**_NOTE:_** For instance, the adaptive stepsizes in Adam are <img src="https://render.githubusercontent.com/render/math?math=1/(\sqrt{v_t}%2B\epsilon)">, where <img src="https://render.githubusercontent.com/render/math?math=v_t"> is the exponential moving average (EMA) of the squared gradient over iterations.

[//]: # ( <img src="https://render.githubusercontent.com/render/math?math=x_{1,2} = \frac{-b \pm \sqrt{b^2-4ac}}{2b}"> )



## 2) Illustration of the impact of different optimizers on adaptive stepsizes 

<img src="https://github.com/guoqiang-x-zhang/AidaOptimizer/blob/main/imgs/Aida_3stage_overall_mean.png" alt="drawing" width="500"/>
 
 Layerwise average stepsizes of the adaptive stepsizes for the top 10 neural layers when training VGG11 over CIFAR10 for 200 epochs.
The jumps at 100 and 160 epoch in the curves are due to the change of the common stepsize. 


<img src="https://github.com/guoqiang-x-zhang/AidaOptimizer/blob/main/imgs/Aida_3stage_overall_std.png" alt="drawing" width="500"/>

sfda safd;jsa l;sad

## 3) Introducing layerwise stepsizes into Adam and AdaBelief
As a side product, we also propose LAdaBelief and LAdam as extensions of AdaBelief and Adam. Considering LAdaBelief, it is deisgned by introducing layerwsie adaptive stepsizes into AdaBelief by pre-processing. By doing so, LAdaBelief only needs to store one parameter per neural layer for computing the layerwisee adaptive stepsize while AdaBelief needs to store the same number of parameters as the DNN model. Similarly, LAdam is designed in a similar manner as LAdaBelief. It is found that the two new methods produce comparable performance as Adam and AdaBelief for a number of typical DNN tasks (see the experiemtal part of the above paper).  

