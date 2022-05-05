# CSE586 Final Project
>Implementation based on [paper](https://arxiv.org/pdf/2103.13915.pdf)
>
>Link of shrinked dataset: https://drive.google.com/drive/folders/13JhsP77VtiV0r1PhX9Rm_RLtpfYel3FW?usp=sharing
>
>To run the model:
>
>In trainer.py
>
>Change --ucf_data_dir to your ucf101 data directory (UCF101/UCF-101)
>
>Change --ucf_label_dir to your ucf101 label directory (UCF101/TrainTestSplits-RecognitionTask/ucfTrainTestlist)
>
>Change --log_dir to your model log directory
>
>Change --save_path to your model save path
>
>Change --batch_size based on the gpu memory (batch size of 3 will take 20GB of memory for the STTS model)
>
>Start training with:
>```bash
> python trainer.py
> ```
>To test different architectures:
>
>In stam/transformer_model.py class SpatialTemporalBLock, change aggregate in self.spatial_temporal_encoder and self.temporal_spatial_encoder to:
>
>temporal_aggregate, spatial_aggregate, temporal_mean_aggregate, spatial_mean_aggregate
>
