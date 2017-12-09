from __future__ import with_statement
import sys
try:
    import queue
except ImportError:
    import Queue as queue
import random
import Pyro4.core
from workitem import Workitem
import os
import socket
from time import sleep, time
from functools import partial

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

CLIENT_NAME = 'C%d@%s:%s' % (os.getpid(), socket.gethostname(), str(time()))
WAIT_DISPATCHER_TIMEOUT = 4

def repeater(function, args=None, kwargs=None, exceptions=Exception,
                     timeout=0):
    args = args or ()
    kwargs = kwargs or {}

    while True:
        try:
            result = function(*args, **kwargs)
            return result
        except exceptions as e:
            print("Error:", e)

            if timeout > 0:
                sleep(timeout)

dispatcherRepeater = partial(
    repeater,
    exceptions=(Pyro4.errors.CommunicationError),
    timeout=WAIT_DISPATCHER_TIMEOUT)

def readNumbers(path):
    print('\nReading numbers')
    with open(path) as f:
        lines = f.read().splitlines()
    numbers = [int(e) for e in lines]
    return numbers


def placeWork(dispatcher, numbers):
    print('\nPlacing work items into dispatcher queue')
    for i in range(len(numbers)):
        item = Workitem(i + 1, CLIENT_NAME, numbers[i])
        dispatcherRepeater(dispatcher.putWork, [item])


def collectResults(dispatcher, item_count):
    print('\nGetting results from dispatcher queue')
    results = {}
    while len(results) < item_count:
        try:
            item = dispatcherRepeater(dispatcher.getResult, [CLIENT_NAME])
            print('Got result: %s (from %s)' % (item, item.processedBy))
            results[item.data] = item.result
        except queue.Empty:
            result_queue_size = dispatcherRepeater(
                dispatcher.resultQueueSize, [CLIENT_NAME])
            print('Not all results available yet (got %d out of %d). Work queue size: %d' %
                  (len(results), item_count, item_count - result_queue_size))

    dispatcherRepeater(dispatcher.clientExit, [CLIENT_NAME])
    return results


def writeResults(results, path):
    print('\nWriting results')
    with open(path, 'w') as f:
        for (number, factorials) in results.items():
            f.write(str(number) + ': ' + ', '.join(map(str, factorials)) + '\n')


def main():
    disp_address = str(sys.argv[1])
    numbers_path = str(sys.argv[2])
    results_path = str(sys.argv[3])

    numbers = readNumbers(numbers_path)

    with Pyro4.core.Proxy('PYRO:dispatcher@' + disp_address) as dispatcher:
        dispatcherRepeater(dispatcher.clientRegister, [CLIENT_NAME])
        placeWork(dispatcher, numbers)
        results = collectResults(dispatcher, len(numbers))

    writeResults(results, results_path)


if __name__ == '__main__':
    main()
