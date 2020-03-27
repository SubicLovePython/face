import yaml


class AttriDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(self, '__key', kwargs.pop('__key', None))
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key,val in arg.items():
                    self[key] = self._hook(val)
            elif isinstance(arg, tuple) and (not isinstance(arg[0], tuple)):
                self[arg[0]] = self._hook(arg[1])
            else:
                for key, val in iter(arg):
                    self[key] = self._hook(val)
        for key, val in kwargs.items():
            self[key] = self._hook(val)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __setattr__(self, key, value):
        if hasattr(self.__class__, key):
            raise AttributeError("'Dict' object attribute {0} is read-only".format(key))

    @classmethod
    def _hook(cls, item):
        if isinstance(item, dict):
            return cls(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(cls._hook(elem) for elem in item)
        return item


def cfg_from_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        yaml_cfg = AttriDict(yaml.load(f))
    return yaml_cfg


if __name__ == '__main__':
    # c = cfg_from_file("../cfgs/ori.yaml")
    # print(c.TRAIN.OPTIMIZATION)
    with open('../cfgs/ori.yaml', 'r') as f:
        config = yaml.load(f.read())
    c = AttriDict(config)
    print(c.TRAIN.OPTIMIZATION.SCHEDULER.TYPE)
