# BalanceSVFND
Code for paper: ['BalanceSVFND: Balanced Multimodal Learning for Fake News Detection on Short Video Platforms']
## Environment
Please refer to the file requirements.txt.
## Dataset
We conduct experiments on two datasets: FakeSV and FakeTT. 
### FakeSV
FakeSV is the largest publicly available Chinese dataset for fake news detection on short video platforms, featuring samples from Douyin and Kuaishou, two popular Chinese short video platforms. 
### FakeTT
FakeTT collect news videos from the TikTok platform, following a similar collection process as FakeSV, provides video, audio and textual descriptions (titles).
## Data Preprocess
- For FakeTT dataset, we use [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to extract OCR.
- Pretrained bert-wwm can be downloaded [here](https://drive.google.com/file/d/1-2vEZfIFCdM1-vJ3GD6DlSyKT4eVXMKq/view), and the folder is already prepared in the project.
- To facilitate reproduction, we provide preprocessed features, which you can download from [this link](https://drive.google.com/file/d/1XKOWxELqsNj9TXBjBPsstgEAPhoeIDM6/view?usp=sharing), Please place these features in the specified location, which can be customized in dataloader.py. 
## Train
After placing the data, start training the model:
```python
python main.py
```

All of our data is shared through anonymous Google Drive accounts. The original dataset can be applied for [FakeSV](https://github.com/ICTMCG/FakeSV) and [FakeTT](https://github.com/ICTMCG/FakingRecipe?tab=readme-ov-file).
