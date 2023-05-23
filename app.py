from flask import Flask, jsonify, request, send_file, make_response
from flask_cors import cross_origin, CORS
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from keras.models import load_model
from keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input
import numpy as np
from werkzeug.utils import secure_filename
import os
import base64

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

from tensorflow.keras.models import Model



def GradCam(model, img_array, layer_name, eps=1e-8):
    '''
    Creates a grad-cam heatmap given a model and a layer name contained with that model
    

    Args:
      model: tf model
      img_array: (img_width x img_width) numpy array
      layer_name: str


    Returns 
      uint8 numpy array with shape (img_height, img_width)

    '''

    gradModel = Model(
			inputs=[model.inputs],
			outputs=[model.get_layer(layer_name).output,
				model.output])
    
    with tf.GradientTape() as tape:
			# cast the image tensor to a float-32 data type, pass the
			# image through the gradient model, and grab the loss
			# associated with the specific class index
      inputs = tf.cast(img_array, tf.float32)
      (convOutputs, predictions) = gradModel(inputs)
      loss = predictions[:, 0]
		# use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)
    
    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
		# the convolution and guided gradients have a batch dimension
		# (which we don't need) so let's grab the volume itself and
		# discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]
    # compute the average of the gradient values, and using them
		# as weights, compute the ponderation of the filters with
		# respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
  
    # grab the spatial dimensions of the input image and resize
		# the output class activation map to match the input image
		# dimensions
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
		# normalize the heatmap such that all values lie in the range
		# [0, 1], scale the resulting values to the range [0, 255],
		# and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    # heatmap = (heatmap * 255).astype("uint8")
		# return the resulting heatmap to the calling function
    return heatmap


def sigmoid(x, a, b, c):
    return c / (1 + np.exp(-a * (x-b)))

def superimpose(img_bgr, cam, thresh, emphasize=False):
    
    '''
    Superimposes a grad-cam heatmap onto an image for model interpretation and visualization.
    

    Args:
      image: (img_width x img_height x 3) numpy array
      grad-cam heatmap: (img_width x img_width) numpy array
      threshold: float
      emphasize: boolean

    Returns 
      uint8 numpy array with shape (img_height, img_width, 3)

    '''
    heatmap = cv2.resize(cam, (img_bgr.shape[1], img_bgr.shape[0]))
    if emphasize:
        heatmap = sigmoid(heatmap, 50, thresh, 1)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    hif = .8
    superimposed_img = heatmap * hif + img_bgr
    superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    
    return superimposed_img



@app.route('/diagnose', methods=['POST'])
@cross_origin(origin='*')
def diagnose():
    file = request.files['file']
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    model = load_model('densenet-09-0.922772-0.932250.h5', compile=False)

    # Compile the model with desired optimizer settings
    optimizer = Adam(lr=0.001, decay=0.0) # specify optimizer and its settings
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Load and preprocess the image
    img_path = UPLOAD_FOLDER + filename
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make prediction
    preds = model.predict(img)
    print(preds)
    class_names = ['COVID-19', 'Non-COVID Disease', 'Normal']
    predicted_class = np.argmax(preds)
    image_output_class=class_names[predicted_class]
    accuracy = np.max(preds)
    accuracy = float(accuracy)

    layer_name = 'conv5_block16_2_conv'
    grad_cam=GradCam(model, img ,layer_name)
    grad_cam_superimposed = superimpose(image.img_to_array(image.load_img(img_path, target_size=(224, 224))), grad_cam, 0.5, emphasize=True)
    

    # Save the Grad-CAM visualization
    gradcam_filename = 'gradcam_' + filename
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    cv2.imwrite(gradcam_path, grad_cam_superimposed)
    with open(gradcam_path, 'rb') as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')
    # image_data = send_file(gradcam_path, mimetype = 'image/png')
    data = {'pred': image_output_class, 'image_data': image_data, 'accuracy': accuracy}
    return jsonify(data)



@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)