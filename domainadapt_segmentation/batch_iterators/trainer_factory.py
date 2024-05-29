
class TrainerRegister: 
    __data = {}
    @classmethod
    def register(cls,cls_name=None):
        def decorator(cls_obj):
            cls.__data[cls_name]=cls_obj
            return cls_obj
        return decorator
    @classmethod
    def get_model(cls,key):
        return cls.__data[key]
    @classmethod
    def num_models(cls):
        return len(cls.__data)
    @classmethod
    def get_models(cls):
        return cls.__data.keys()
def load_trainer(trainer_name): 
    return TrainerRegister.get_trainer(trainer_name)