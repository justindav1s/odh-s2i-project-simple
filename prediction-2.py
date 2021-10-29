# For sample predict functions for popular libraries visit https://github.com/opendatahub-io/odh-prediction-samples

# Import libraries
# import tensorflow as tf

# Load your model.
# model_dir = 'models/myfancymodel'
# saved_model = tf.saved_model.load(model_dir)
# predictor = saved_model.signatures['default']

from allcnn_predict_local import predictImage
from allcnn_predict_local import processImage

# Write a predict function 
def predict():
#     arg = args_dict.get('arg1')
#     predictor(arg)
    categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    url = "https://jndfiles-pub.s3.eu-west-1.amazonaws.com/images/dogs/dogs-8.jpg"
    img = processImage(url)
    best_prob, best_index, sec_best_prob, sec_best_index, thd_best_prob, thd_best_index = predictImage(img)
    print ("1st prediction : "+str(categories[best_index])+" prob : "+'{:05.3f}'.format(best_prob))
    print ("2nd prediction : "+str(categories[sec_best_index])+" prob : "+'{:05.3f}'.format(sec_best_prob))
    print ("3nd prediction : "+str(categories[thd_best_index])+" prob : "+'{:05.3f}'.format(thd_best_prob))
    return {'prediction': str(categories[best_index])}


predict()

