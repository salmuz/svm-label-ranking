# Authors: Yonatan-Carlos Carranza-Alarcon <salmuz@gmail.com>
# First version source in:
#  (1) https://github.com/salmuz/classifip/blob/master/experiments/classification/mlc/multitask.py
#  (2) https://github.com/salmuz/classifip/blob/master/experiments/classification/qda/qda_manager.py
# License: BSD 3 clause

from multiprocessing import Process, Queue, cpu_count, JoinableQueue
import importlib
import time


def __create_dynamic_class(clazz):
    try:
        st = clazz.split(".")
        module_name = ".".join(st[:-1])
        class_name = st[-1]
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_()
    except Exception as e:
        raise Exception(e, "Creation dynamic class does not complete.")


def prediction(pid, tasks, queue, results, class_model):
    try:
        model = __create_dynamic_class(class_model)
        while True:
            training = queue.get()
            if training is None:
                break
            model.learn(**training)
            while True:
                task = tasks.get()
                if task is None:
                    break
                prediction = model.evaluate(**task['kwargs'])
                print(time.strftime('%x %X %Z'), "(pid, prediction, ground-truth) ",
                      pid, prediction, task['y_test'], flush=True)
                results.put(dict({'prediction': prediction, 'ground_truth': task['y_test']}))
            queue.task_done()
    except Exception as e:
        raise Exception(e, "Error in job of PID " + pid)
    finally:
        print("Worker PID finished", pid, flush=True)


class ManagerWorkers:

    def __init__(self, nb_process):
        self.workers = None
        self.tasks = Queue()
        self.results = Queue()
        self.qeTraining = [JoinableQueue() for _ in range(nb_process)]
        self.NUMBER_OF_PROCESSES = cpu_count() if nb_process is None else nb_process

    def executeAsync(self, class_model):
        print("Starting %d workers" % self.NUMBER_OF_PROCESSES, flush=True)
        self.workers = []
        for i in range(self.NUMBER_OF_PROCESSES):
            p = Process(target=prediction,
                        args=(i, self.tasks, self.qeTraining[i], self.results, class_model,))
            self.workers.append(p)

        for w in self.workers:
            w.start()

    def addNewTraining(self, **kwargs):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].put(kwargs)

    def poisonPillTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].put(None)

    def joinTraining(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.qeTraining[i].join()

    def addTask(self, task):
        self.tasks.put(task)

    def waitWorkers(self):
        for w in self.workers:
            w.join()

    def getResults(self):
        return self.results

    def poisonPillWorkers(self):
        for i in range(self.NUMBER_OF_PROCESSES):
            self.addTask(None)
