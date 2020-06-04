class LazyList(list):
    "Like a list, but invokes callables"
    def __getitem__(self, key):
        item = super().__getitem__(key)
        if callable(item):
            item = item()
        return item
