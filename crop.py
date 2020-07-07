from PIL import Image
from torchvision import transforms
import os

folds = os.listdir("./dataset/")
cnt = 0
for f in folds:

	if f.startswith("set"):
		cnt = cnt + 1
		print(f)
		os.mkdir("./transformed_CAL/"+f)
		os.mkdir("./transformed_CAL/"+f+"/CameraCenter_Images/")
		ims = os.listdir("./dataset/"+f+"/CameraCenter_Images/")
		for i in ims:
			im = Image.open("./dataset/"+f+"/CameraCenter_Images/"+i)
			cropped = im.crop((0,120,800,480))
			w, h = [int(s*0.4) for s in cropped.size]
        		resized = transforms.Resize((h, w))(cropped)
        		#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        		#normed = normalize(transforms.functional.to_tensor(resized))
			resized.save("./transformed_CAL/"+f+"/CameraCenter_Images/"+i)

		os.mkdir("./transformed_CAL/"+f+"/CameraLeft_Images/")
		ims = os.listdir("./dataset/"+f+"/CameraLeft_Images/")
		for i in ims:
			im = Image.open("./dataset/"+f+"/CameraLeft_Images/"+i)
			cropped = im.crop((0,120,800,480))
			w, h = [int(s*0.4) for s in cropped.size]
        		resized = transforms.Resize((h, w))(cropped)
        		#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        		#normed = normalize(transforms.functional.to_tensor(resized))
			resized.save("./transformed_CAL/"+f+"/CameraLeft_Images/"+i)

		os.mkdir("./transformed_CAL/"+f+"/CameraRight_Images/")
		ims = os.listdir("./dataset/"+f+"/CameraRight_Images/")
		for i in ims:
			im = Image.open("./dataset/"+f+"/CameraRight_Images/"+i)
			cropped = im.crop((0,120,800,480))
			w, h = [int(s*0.4) for s in cropped.size]
        		resized = transforms.Resize((h, w))(cropped)
        		#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        		#normed = normalize(transforms.functional.to_tensor(resized))
			resized.save("./transformed_CAL/"+f+"/CameraRight_Images/"+i)
		print(cnt)
		print("Done")
		#break

