!pip install  watson-machine-learning-client
Collecting watson-machine-learning-client
  Downloading watson_machine_learning_client-1.0.391-py3-none-any.whl (538 kB)
Requirement already satisfied: boto3 in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (1.21.32)
Requirement already satisfied: tqdm in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (4.64.0)
Requirement already satisfied: pandas in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (1.4.2)
Requirement already satisfied: tabulate in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (0.8.9)
Requirement already satisfied: ibm-cos-sdk in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (2.11.0)
Requirement already satisfied: requests in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (2.27.1)
Requirement already satisfied: certifi in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (2021.10.8)
Requirement already satisfied: urllib3 in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (1.26.9)
Requirement already satisfied: lomond in c:\users\karup\anaconda3\lib\site-packages (from watson-machine-learning-client) (0.3.3)
Requirement already satisfied: botocore<1.25.0,>=1.24.32 in c:\users\karup\anaconda3\lib\site-packages (from boto3->watson-machine-learning-client) (1.24.32)
Requirement already satisfied: s3transfer<0.6.0,>=0.5.0 in c:\users\karup\anaconda3\lib\site-packages (from boto3->watson-machine-learning-client) (0.5.0)
Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in c:\users\karup\anaconda3\lib\site-packages (from boto3->watson-machine-learning-client) (0.10.0)
Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in c:\users\karup\anaconda3\lib\site-packages (from botocore<1.25.0,>=1.24.32->boto3->watson-machine-learning-client) (2.8.2)
Requirement already satisfied: six>=1.5 in c:\users\karup\anaconda3\lib\site-packages (from python-dateutil<3.0.0,>=2.1->botocore<1.25.0,>=1.24.32->boto3->watson-machine-learning-client) (1.16.0)
Requirement already satisfied: ibm-cos-sdk-core==2.11.0 in c:\users\karup\anaconda3\lib\site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
Requirement already satisfied: ibm-cos-sdk-s3transfer==2.11.0 in c:\users\karup\anaconda3\lib\site-packages (from ibm-cos-sdk->watson-machine-learning-client) (2.11.0)
Requirement already satisfied: charset-normalizer~=2.0.0 in c:\users\karup\anaconda3\lib\site-packages (from requests->watson-machine-learning-client) (2.0.4)
Requirement already satisfied: idna<4,>=2.5 in c:\users\karup\anaconda3\lib\site-packages (from requests->watson-machine-learning-client) (3.3)
Requirement already satisfied: numpy>=1.18.5 in c:\users\karup\anaconda3\lib\site-packages (from pandas->watson-machine-learning-client) (1.21.5)
Requirement already satisfied: pytz>=2020.1 in c:\users\karup\anaconda3\lib\site-packages (from pandas->watson-machine-learning-client) (2021.3)
Requirement already satisfied: colorama in c:\users\karup\anaconda3\lib\site-packages (from tqdm->watson-machine-learning-client) (0.4.4)
Installing collected packages: watson-machine-learning-client
Successfully installed watson-machine-learning-client-1.0.391
from ibm_watson_machine_learning import APIClient
wml_credentials={
    "url":"https://us-south.ml.cloud.ibm.com",
    "apikey":"BPFGcOrCf3sroRy3uKOPGozsmIL-5oVDv4A_lru2IpMS"
    
}

client=APIClient(wml_credentials)
client=APIClient(wml_credentials)
client
 def guid_from_space_name(client, space_name):
     space=client.spaces.get_details()
     #print(space)
     return(next(item for item in space['resources'] if item['entity']['name']== space_name)['metadata']['id'])
space_uid=guid_from_space_name(client,'imageclassification') #imageclassification is the deployment space name
print("Space UID ="+space_uid)
Space UID =ba02adea-7e10-4237-81e7-eaf084fe4102
client.set.default_space(space_uid)
'SUCCESS'
client.repository.download("f3e12114-24f4-4bae-9d60-2897d27e7ce6", 'nutrition.tar.gz')
Successfully saved model content to file: 'nutrition.tar.gz'
'C:\\Users\\karup\\Downloads/nutrition.tar.gz'
from keras.models import load_model
from keras.preprocessing import image
model = load_model("nutrition.h5")
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
#loading of the image
img = load_img("apple.jpg", grayscale=False,target_size=(64,64))
#image to array 
x = img_to_array(img)
#changing the shape
x= np.expand_dims(x,axis = 0)
predict_x=model.predict(x)
classes_x=np.argmax(predict_x,axis = -1)
classes_x
1/1 [==============================] - 1s 696ms/step
array([0], dtype=int64)
index=['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']
result=str(index[classes_x[0]])
result
'APPLES'
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
#loading of the image
img = load_img("banana.jpg", grayscale=False,target_size=(64,64))
#image to array 
x = img_to_array(img)
#changing the shape
x= np.expand_dims(x,axis = 0)
predict_x=model.predict(x)
classes_x=np.argmax(predict_x,axis = -1)
classes_x
1/1 [==============================] - 0s 40ms/step
array([1], dtype=int64)
index=['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']
result=str(index[classes_x[0]])
result
'BANANA'
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
#loading of the image
img = load_img("Test_Image5.jpg", grayscale=False,target_size=(64,64))
#image to array 
x = img_to_array(img)
#changing the shape
x= np.expand_dims(x,axis = 0)
predict_x=model.predict(x)
classes_x=np.argmax(predict_x,axis = -1)
classes_x
1/1 [==============================] - 0s 40ms/step
array([3], dtype=int64)
index=['APPLES', 'BANANA', 'ORANGE', 'PINEAPPLE', 'WATERMELON']
result=str(index[classes_x[0]])
result
'PINEAPPLE'