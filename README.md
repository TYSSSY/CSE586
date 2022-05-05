# CSE586 Final Project
>Implementation based on [paper](https://arxiv.org/pdf/2103.13915.pdf)

>To run the model:
>In trainer.py
>Change --ucf_data_dir to your ucf101 data directory (UCF101/UCF-101)
>Change --ucf_label_dir to your ucf101 label directory (UCF101/TrainTestSplits-RecognitionTask/ucfTrainTestlist)
>Change --log_dir to your model log directory
>Change --save_path to your model save path
>Change --batch_size based on the gpu memory (batch size of 3 will take 20GB of memory for the STTS model)

>Start training with:
>```bash
> python trainer.py
> ```
