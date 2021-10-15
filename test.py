from model import *
from data import *
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model = unet()
model.load_weights("trained_unet_model.hdf5")

cwd=os.getcwd()
test_dir="testing_images2"

abs_testing_image_dir=os.path.join(os.sep,cwd,test_dir)


for f in os.listdir(abs_testing_image_dir):
    if re.search("predict", f):
        os.remove(os.path.join(abs_testing_image_dir, f))

    if re.search("Thumbs", f):
        os.remove("Thumbs.db")


num_files=len(os.listdir(abs_testing_image_dir))

testGene = testGenerator(abs_testing_image_dir,num_image=num_files)
results = model.predict_generator(testGene,num_files,verbose=1)

#results will be saved in same directory as testing images
saveResult(abs_testing_image_dir,results)


   


