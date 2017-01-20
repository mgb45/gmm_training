# gmm_training

Mixture of random walk motion model training for upper body pose estimation using [mkfbodytracker_pdaf](https://github.com/mgb45/mkfbodytracker_pdaf). For more info see 

> Burke, M. G. (2015). Fast upper body pose estimation for human-robot interaction (doctoral thesis). https://doi.org/10.17863/CAM.203

* Data for training needs to be formatted as in KinectJoints.yaml (ordered hand x,y,z elbow x,y,z, shoulder x,y,z, head x,y,z, neck x,y,z, shoulder x,y,z, elbow x,y,z hand x,y,z)
* Clone repo into a rosbuild workspace

```
rosws set gmm_training
cd gmm_training
rosmake gmm_training
rosrun gmm_training buildModel3D_PCA KinectJoints.yaml #samples #clusters model_dims
```

* The source is hand tuned with a camera calibration matrix for a Kinect. If you are using another camera, you will need to modify this, recompile and retrain.
* Training produces two files for use with [mkfbodytracker_pdaf](https://github.com/mgb45/mkfbodytracker_pdaf), one for each arm, with the naming convention 
```
data13D_PCA_<#samples>_<#clusters>_<model_dims>.yml
data23D_PCA_<#samples>_<#clusters>_<model_dims>.yml
```
