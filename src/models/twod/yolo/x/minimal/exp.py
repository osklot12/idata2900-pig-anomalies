from yolox.exp import Exp as BaseExp

class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.input_size = (640, 640)
        self.test_size = (640, 640)
        self.data_num_workers = 0
        self.max_epoch = 1
        self.print_interval = 1
        self.eval_interval = 1
        self.exp_name = "dummy_exp"