from __future__ import print_function
import sys

try:
    import queue
except ImportError:
    import Queue as queue
import Pyro4.core

from collections import defaultdict
from itertools import count
from time import sleep, time
import threading

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

HEARTBEAT_STEP = 1
HEARTBEAT_MAX_DELAY = 4


class DispatcherQueue(object):
    def __init__(self):
        self.workqueue = queue.Queue()
        self.activeworks = defaultdict(queue.Queue)
        self.processeditems = defaultdict(defaultdict)
        self.resultqueues = defaultdict(queue.Queue)
        self.heartbeats = defaultdict(int)
        self.lock = threading.Lock()
        Pyro4.futures.Future(self.monitorHartbeats)()

    @Pyro4.expose
    def putWork(self, item):
        self.workqueue.put(item)

    @Pyro4.expose
    def getWork(self, worker):
        item = self.workqueue.get(timeout=0)
        self.activeworks[worker] = item
        return item

    @Pyro4.expose
    def putResult(self, worker, item):
        processeditems = self.processeditems.get(item.client, None)
        if processeditems is None:
            return

        clientqueue = self.resultqueues.get(item.client, None)
        if clientqueue is None:
            return

        cur_item = processeditems.setdefault(item.itemId, item)
        if item is cur_item: # set item as new processed item
            clientqueue.put(item)

        self.activeworks.pop(worker, None)

    @Pyro4.expose
    def getResult(self, client, timeout=5):
        return self.resultqueues[client].get(timeout=timeout)

    @Pyro4.expose
    def resultQueueSize(self, client):
        return self.resultqueues[client].qsize()

    @Pyro4.expose
    def clientRegister(self, client):
        self.processeditems[client]
        self.resultqueues[client]

    @Pyro4.expose
    def clientExit(self, client):
        self.processeditems.pop(client, None)
        self.resultqueues.pop(client, None)

    @Pyro4.expose
    def workerRegister(self, worker):
        print('Register worker:', worker)
        self.heartbeats[worker] = time()

    @Pyro4.expose
    def updateHeartbeat(self, worker):
        self.heartbeats[worker] = time() # TODO: just if time > old_time

    def monitorHartbeats(self):
        while True:
            sleep(HEARTBEAT_STEP)
            workers = self.heartbeats.keys()
            for worker in workers:
                timestemp = self.heartbeats.get(worker, None)
                if timestemp is None:
                    continue

                if time() - timestemp > HEARTBEAT_MAX_DELAY:
                    item = self.activeworks.get(worker, None)
                    if item is not None:
                        self.putWork(self.activeworks[worker])

                    self.lock.acquire()
                    if time() - timestemp:
                        self.activeworks.pop(worker, None)
                        self.heartbeats.pop(worker, None)
                    self.lock.release()

                    print('Oops, worker', worker, 'closed')



def main():
    # HOST:PORT
    address = str(sys.argv[1]).split(':')
    host = address[0]
    port = int(address[1])

    daemon = Pyro4.core.Daemon(host, port)

    dispatcher = DispatcherQueue()

    uri = daemon.register(dispatcher, 'dispatcher')
    print('Dispatcher is running: ' + str(uri))
    daemon.requestLoop()


if __name__ == '__main__':
    main()
