class Workitem(object):
    def __init__(self, itemId, client, data):
        print('Created workitem %s' % itemId)
        self.itemId = itemId
        self.client = client
        self.data = data
        self.result = None
        self.processedBy = None

    def __str__(self):
        return '<Workitem %s %d>' % (self.client, self.itemId)
