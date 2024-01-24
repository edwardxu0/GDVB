class PLOT:
    def __init__(self):
        self

    def draw(self):
        pass

    def save(self, path):
        self.plt.savefig(path, bbox_inches='tight',pad_inches=0)
        self.plt.close()
