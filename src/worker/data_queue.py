import multiprocessing

# Shared queue between receiver and trainer
data_queue = multiprocessing.Queue(maxsize=100)
