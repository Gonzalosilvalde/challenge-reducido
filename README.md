[dataset](https://drive.google.com/drive/folders/10KqkaOAGKKdYK6qrFce3chwW7sIeyDWS)
	we are loading all every time, not really efficient
	[Zindi](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge)
		16x16 pixel patch extracted from a Landsat image. Each pixel patch has 6 spectral bands: Blue, Green, Red, Near-infrared (NIR), Shortwave infrared (SWIR1) Shortwave infrared 2 (SWIR2)
		30 meters per pixel. Each 16x16 patch likely represents a 480m x 480m area 
	[(PDF) EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification](https://www.researchgate.net/publication/319463676_EuroSAT_A_Novel_Dataset_and_Deep_Learning_Benchmark_for_Land_Use_and_Land_Cover_Classification#pf9)
			- Start with a pretrained ResNet-50 (or other architecture) model. 
			- Replace the final classification layer to match your number of classes.
			- Freeze all layers except the final layer and train it with a relatively high learning rate (e.g., 0.01).
			- Unfreeze all layers and fine-tune the entire network with a lower learning rate (e.g., 0.001 or 0.0001).
    landsat pretrained? maybe stupid resolution of images too different... ( SSL4EO-L dataset are 264 × 264)
        https://github.com/allenai/satlaspretrain_models?tab=readme-ov-file
        https://www.arcgis.com/home/item.html?id=e732ee81a9c14c238a14df554a8e3225
        https://torchgeo.readthedocs.io/en/stable/api/models.html#landsat  
            ResNet50_Weights.LANDSAT_ETM_SR_MOCO looks decent
	upscale sat images?
	[satellite-image-deep-learning · GitHub](https://github.com/satellite-image-deep-learning)
	dataset augmentation?
		[Albumentations: fast and flexible image augmentations](https://albumentations.ai/)
		active learning, see which give the most info
	pretrained vision transformers? prob bad
		[ibm-nasa-geospatial/Prithvi-100M · Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)
		[GitHub - nasaharvest/presto: Lightweight, Pre-trained Transformers for Remote Sensing Timeseries](https://github.com/nasaharvest/presto)
	Class Imbalance: If settlements are rare in your dataset, consider using weighted loss or oversampling techniques.
		!!!! Number of 0s: 1000000 (90.91%), Number of 1s: 100000 (9.09%)
		weighted cross entropy ?
	ENSEMBLE
	[Land use land cover image classification using deep learning | EuroSat | ResNet50 | GeoDev - YouTube](https://youtu.be/5BNHcLDeirs?t=691)
	refractor to pytorch lignhtning?
	train on validation!
