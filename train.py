from options import Options
from trainer import Trainer

options = Options()
opts = options.parse()

if __name__=="__main__":
    trainer = Trainer(opts)
    trainer.train()
