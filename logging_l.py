import sys
import os
import datetime
def make_print_to_file(path='./'):
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            if not os.path.exists(path):
                os.mkdir(path)
            self.log = open(os.path.join(path, filename), "a+", encoding='utf8',)

 
        def write(self, message):

            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
    #fileName = datetime.datetime.now().strftime('log_'+'%Y_%m_%d')
    fileName = datetime.datetime.now().strftime('log_'+'%Y_%m_%d_%H_%M')
    sys.stdout = Logger(fileName + '.log', path=path)

if __name__ == '__main__': 
    make_print_to_file(path='./result/log')

