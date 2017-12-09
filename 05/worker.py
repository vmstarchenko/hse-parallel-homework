from __future__ import print_function
import os
import socket
import sys
from math import floor, sqrt
from time import time, sleep
from multiprocessing import Pool, TimeoutError
from contextlib import closing

try:
    import queue
except ImportError:
    import Queue as queue

import Pyro4.core
from workitem import Workitem


Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

WORKERNAME = 'Worker_%d@%s:%s' % (os.getpid(), socket.gethostname(), str(time()))

WAIT_DISPATCHER_TIMEOUT = 4
MAIN_LOOP_SLEEP_TIME = 0.1
HEARTBEAT_STEP = 1

long = int


def factorize(n):
    def step(x): return 1 + (x << 2) - ((x >> 1) << 1)
    maxq = long(floor(sqrt(n)))
    d = 1
    q = n % 2 == 0 and 2 or 3
    while q <= maxq and n % q != 0:
        q = step(d)
        d += 1
    return q <= maxq and [q] + factorize(n // q) or [n]


def process(item):
    print('factorizing %s -->' % item.data)
    sys.stdout.flush()
    item.result = factorize(int(item.data))
    print(item.result)
    item.processedBy = WORKERNAME
    return item

def heartbeat(dispatcher):
    
    def updateHeartbeat():
        cur = time()
        if cur - updateHeartbeat.prev > HEARTBEAT_STEP:
            dispatcher.updateHeartbeat(WORKERNAME)
            updateHeartbeat.prev = cur

    updateHeartbeat.prev = time()

    return updateHeartbeat

def workerLoop(dispatcher, pool):
    worker = None
    busy = False
    updateHeartbeat = heartbeat(dispatcher)
    while True:
        updateHeartbeat()
        if not busy:
            try:
                item = dispatcher.getWork(WORKERNAME)
            except queue.Empty:
                print('no work available yet')
            else:
                busy = True
                worker = pool.apply_async(process, (item,))
                      # runs in *only* one process

        elif worker is not None and worker.ready():
            item = worker.get()
            dispatcher.putResult(WORKERNAME, item)
            busy = False
            worker = None

        sleep(MAIN_LOOP_SLEEP_TIME)

        # Pyro4.errors.ConnectionClosedError

def register(dispatcher):
    while True:
        try:
            dispatcher.workerRegister(WORKERNAME)
            break
        except Pyro4.errors.CommunicationError:
            print('Can\'t connect to dispatcher')
            sleep(WAIT_DISPATCHER_TIMEOUT)

def main():
    disp_address = str(sys.argv[1])
    dispatcher = Pyro4.core.Proxy('PYRO:dispatcher@' + disp_address)
    print('This is worker %s' % WORKERNAME)

    with closing(Pool(processes=1)) as pool:
        while True:
            try:
                register(dispatcher)
                workerLoop(dispatcher, pool)
            except Exception as e:
                print('Oops:', e)
                sleep(WAIT_DISPATCHER_TIMEOUT)


if __name__ == '__main__':
    main()
