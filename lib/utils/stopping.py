

class ES(object):
    '''
    Early stopping is a technique used to stop training the neural network at the right time.
    '''

    def __init__(self, patience=2, mode='max'):
        self.check = "0" if mode == 'max' else "np.inf"
        self.mode = mode
        self.patience = patience

        # count the number of times metric hasn't improved
        self.p = 0

    def continue_training(self, metric_value):
        # this is a make shift function, do suggest yours in the comments
        # feel free to fork and change it if you wish to
        curr = eval(f"{self.mode}({self.check}, {metric_value})")
        if self.check != curr:
            self.p = 0
            self.check = curr
            return True
        else:
            self.p += 1
            return False if self.p >= self.patience else True
