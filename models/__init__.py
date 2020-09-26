#from models.CNN_models.vgg import VGG16
#from models.CNN_models.ResNet18 import ResNet18
from models.CNN_models.lenet import LeNet
#from models.CNN_modles.name_of_file_import name_of_class


def get_network(name: str) -> None:
    return \
        LeNet() if name == 'LeNet' else\
        #VGG16() if name == 'VGG16' else\
        #ResNet18() if name == 'ResNet18' else\
        None
