from itertools import islice, chain, tee
from collections import deque
from sortedcontainers import SortedList
from pathlib import Path
from functools import reduce
from typing import get_type_hints
from typing import Optional, Iterable, Any, List, Tuple, NamedTuple, Callable, Dict, Literal
import json
import cv2
import csv
import re
from tqdm import tqdm

from ..check import NDArray, SizeVar
from ..io import images_to_batch, batch_to_masks
import numpy as np


def grouper(iterable: Iterable[Any], n: int):
    """
    Divides sequence to batches of n elements
    :param n: batch size
    :param iterable: input sequence
    :return: batches
    """
    it = iter(iterable)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            return
        yield chunk


class PipelineItem(NamedTuple):
    prev_idx: Tuple[int, ...]
    names: Dict[str, Any]
    code: str
    out: Optional[str]
    batch_shape: Tuple[int, ...] = ()
    type: Literal['func', 'cull', 'source', 'end'] = 'func'
    need_frame_no: bool = False
    out_is_tuple: bool = False


class MaxBufferItem(NamedTuple):
    frame_no: int
    value: float
    data: Optional[Any]


class MaxQueue(deque):
    def __init__(self, filter_size: int):
        super(MaxQueue, self).__init__(maxlen=filter_size + 1)
        self.half_size = filter_size // 2
        self.last: Optional[int] = None

    def push(self, item: MaxBufferItem) -> Optional[MaxBufferItem]:
        # Put item
        while (
                len(self) > 0
                and item.frame_no - self[-1].frame_no <= self.half_size
                and item.value > self[-1].value
        ):
            self.pop()
        if (
                len(self) >= 2
                and self[-1].frame_no - self[-2].frame_no <= self.half_size
                and item.value == self[-1].value
        ):
            self.pop()
        self.append(item)

        # Get item:
        return self.get(item.frame_no)

    def get(self, current: Optional[int] = None) -> Optional[MaxBufferItem]:
        if len(self) > 0 and (current is None or current - self[0].frame_no > self.half_size):
            item = self.popleft()
            last = self.last
            self.last = item.frame_no
            if last is None or item.frame_no - last:
                return item
        else:
            return None


class MaxBuffer:
    def __init__(self, size: int, filter_size: int):
        self.size = size
        self.data = SortedList(key=lambda v: -v.value)
        self.queue = MaxQueue(filter_size)
        self.worst = None
        self.filter_size = filter_size
        self.last_item: Optional[MaxBufferItem] = None

    def append(self, value: float, frame_no: int, data: Any) -> None:
        filtered = self.queue.push(MaxBufferItem(frame_no, value, data))
        if filtered is not None:
            self.data.add(filtered)
            if len(self.data) > self.size:
                self.data.pop(0)


class Dumper:
    types_map = {
        float: np.float32,
        bool: np.bool,
        int: np.int32,
    }

    def __init__(self, name: str, max_thresholds: NamedTuple,
                 max_count: Optional[int] = 5, max_filter: Optional[int] = 100,
                 dumpers=None, names=None):
        self.data = None
        self.max_thresholds = max_thresholds
        self.fields = list(getattr(max_thresholds, '_fields'))
        types = getattr(max_thresholds, '_field_types')
        self.types = list(map(types.get, self.fields))
        self.data_start = 0
        self.filters: List[MaxBuffer] = [
            MaxBuffer(max_count, max_filter) for i in range(len(self.fields))
        ]
        if dumpers[0] is None:
            dumpers[0] = self.dump_metric
        self.dumpers = dumpers
        self.names = names

    @staticmethod
    def dump_metric(data, path, name, frame_no: int):
        def cvt(v):
            if type(v).__module__ == 'numpy':
                return v.item()
            else:
                return v

        data = {k: cvt(v) for k, v in zip(getattr(data, '_fields'), data)}
        data['frame_no'] = frame_no
        with (path / f'{name}.json').open('w') as file:
            json.dump(data, file, indent='  ')

    def on_dump(self, frame_no: int, metric: NamedTuple, *others):
        self.data[frame_no] = (True,) + metric

        for x, ma, mb in zip(metric, self.max_thresholds, self.filters):
            if ma is not None and x > ma:
                mb.append(x - ma, frame_no, (metric,) + others)

        return metric

    def reset(self, data_len: int, data_start: int = 0):
        self.data_start = data_start
        types = list(zip(self.fields, map(self.types_map.get, self.types)))
        self.data = np.zeros(
            (data_len,),
            [('present', np.bool)] + types
        )

    def finalize(self, report_dir: Path):
        report_dir.mkdir(parents=True, exist_ok=True)
        with (report_dir / 'report.csv').open('w') as file:
            writer = csv.writer(file)
            writer.writerow(['frame_no'] + self.fields)
            for frame_no, item in enumerate(self.data):
                if item[0]:
                    row = (frame_no + self.data_start,) + item.item()[1:]
                    writer.writerow(row)
        for name, mb in zip(self.fields, self.filters):
            for i, (frame_no, value, data) in enumerate(mb.data):
                frame_name = f'{name}_{i + 1:02}'
                for pipe_name, dumper, item in zip(self.names, self.dumpers, data):
                    out_name = f'{frame_name}_{pipe_name}'
                    if dumper is None:
                        if (
                                isinstance(item, np.ndarray) and item.dtype == np.uint8 and
                                ((len(item.shape) == 3 and item.shape[-1] == 3) or (len(item.shape) == 2))
                        ):
                            cv2.imwrite(str(report_dir / f'{out_name}.png'), item)
                        else:
                            ...
                    else:
                        dumper(item, report_dir, out_name, frame_no)


class PipelineData:
    def __init__(self, source: Optional[Iterable[Any]] = None):
        self.source = source
        self.dumpers: List[Dumper] = []
        self.steps: List[Optional[PipelineItem]] = [PipelineItem(
            (),
            {},
            'src = tqdm(src)',
            'src',
            (),
            'source'
        )]


class Metric:
    def __init__(self, name: str, pipeline: 'Pipeline'):
        self.name = name
        self.pipeline = pipeline
        self.cull_max_thresholds = None

    def dump(self, *pipelines, name: Optional[str] = None, dumpers=None, **kwargs):
        names = ['metric'] + [p.name_ for p in pipelines]
        dumper = Dumper(
            name or self.name,
            dumpers=dumpers or [None] * (len(pipelines) + 1),
            names=names,
            **kwargs
        )
        self.pipeline.data.dumpers.append(dumper)
        self.pipeline = self.pipeline.stack(*pipelines, need_frame_no=True).map(dumper.on_dump)
        return self

    def cull(self, max_thresholds: NamedTuple):
        if self.cull_max_thresholds is not None:
            raise ValueError('Second cull not allowed.')
        self.cull_max_thresholds = max_thresholds
        self.pipeline = self.pipeline.culling(self.on_cull)
        return self

    def on_cull(self, metric: NamedTuple) -> bool:
        for x, ma in zip(metric, self.cull_max_thresholds):
            if ma is not None and x > ma:
                return False
        return True


class Pipeline:
    def __init__(self, source: Optional[Iterable[Any]] = None, data: Optional[PipelineData] = None):
        if data is None:
            self.data = PipelineData(source)
            self.idx = 0
        else:
            self.data = data
            self.idx = len(self.data.steps)
        self.program = None
        self.name_: Optional[str] = None
        self.names = None
        self.batch_shape: Tuple[int, ...] = ()

    def name(self, name: str) -> 'Pipeline':
        self.name_ = name
        return self

    def stage(self, name: str, pipeline: 'Pipeline'):
        ...

    def out(self):
        return self.data.steps[self.idx].out

    def out_is_tuple(self):
        return self.data.steps[self.idx].out_is_tuple

    def metric(self, func: Callable, name: Optional[str] = None) -> Metric:
        if name is None:
            th = get_type_hints(func)
            name = re.sub(r'(?<!^)(?=[A-Z])', '_', th['return'].__name__).lower()
        return Metric(name, self.map(func, out_prefix='metric_'))

    @staticmethod
    def get_func_name(func):
        name = getattr(func, '__name__', func.__class__.__name__)
        if name == '<lambda>':
            name = 'lambda'
        return name

    def map(self, func, out_prefix: str = '', need_frame_no: bool = False, **kwargs) -> 'Pipeline':
        """
        Apply function to every item
        :param out_prefix:
        :param func: Function to apply
        :param kwargs:
        :return: next Pipeline
        """
        pl = Pipeline(None, self.data)
        pl.batch_shape = self.batch_shape
        name = f'{self.get_func_name(func)}_{pl.idx}'
        out_name = f'{out_prefix}map_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,),
            {name: func},
            f'{out_name} = ({name}({"*" if self.out_is_tuple() else ""}i) for i in {self.out()})',
            out_name,
            self.batch_shape,
            'func',
            need_frame_no
        ))
        return pl

    def segmentation(self, model, batch=None, **kwargs) -> 'Pipeline':
        b = batch or SizeVar('B')
        h = SizeVar('H')
        w = SizeVar('W')
        c = SizeVar('C')

        def wrapper(images: List[NDArray[h, w, 3, np.uint8]]) -> NDArray[b, h, w, c]:
            input_batch = images_to_batch(images, model=model)
            return batch_to_masks(model(input_batch), **kwargs)

        return self.map(wrapper)

    def add(self, *pipelines: 'Pipeline', initial=0) -> 'Pipeline':
        pl = Pipeline(None, self.data)
        args = ', '.join(p.out() for p in pipelines + (self,))
        out_name = f'sum_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,) + tuple(p.idx for p in pipelines),
            {},
            f'{out_name} = map(lambda a: sum(a, {initial}), zip({args}))',
            out_name,
            self.batch_shape
        ))
        pl.previous = [self] + list(pipelines)
        return pl

    def stack(self, *pipelines: 'Pipeline', need_frame_no: bool = False) -> 'Pipeline':
        pl = Pipeline(None, self.data)
        items = [p.out() for p in (self,) + pipelines]
        if need_frame_no:
            items = ['frame_no'] + items
        args = ', '.join(items)
        out_name = f'stack_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,) + tuple(p.idx for p in pipelines),
            {},
            f'{out_name} = map(lambda a: sum(map(lambda i: (i,), a), ()), zip({args}))',
            out_name,
            self.batch_shape,
            'func',
            need_frame_no,
            True
        ))
        pl.previous = [self] + list(pipelines)
        return pl

    def batch(self, size: int) -> 'Pipeline':
        assert size > 0
        pl = Pipeline(None, self.data)
        bs = self.batch_shape + (size,)
        pl.batch_shape = bs
        out_name = f'batch_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,),
            {},
            f'{out_name} = grouper({self.out()}, {size})',
            out_name,
            bs
        ))
        return pl

    def unbatch(self) -> 'Pipeline':
        assert self.batch_shape
        pl = Pipeline(None, self.data)
        bs = self.batch_shape[:-1]
        pl.batch_shape = bs
        out_name = f'unbatch_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,),
            {},
            f'{out_name} = chain.from_iterable({self.out()})',
            out_name,
            bs
        ))
        return pl

    def write(self, func: Callable) -> None:
        self.data.steps.append(PipelineItem(
            (self.idx,),
            {'write': func},
            f'  write({self.out()}_item)',
            None,
            self.batch_shape,
            'end'
        ))

    def culling(self, func: Callable):
        """Breaks pipeline graph: throws away bad frames"""

        pl = Pipeline(None, self.data)
        pl.batch_shape = self.batch_shape
        name = f'{self.get_func_name(func)}_{pl.idx}'
        out_name = f'cull_{pl.idx}'
        self.data.steps.append(PipelineItem(
            (self.idx,),
            {name: func},
            f'{out_name} = map({name}, {self.out()})',
            out_name,
            self.batch_shape,
            'cull'
        ))

        return pl

    @staticmethod
    def _cull(condition, *iterables):
        def gen():
            for c, t in zip(condition, zip(*iterables)):
                if c:
                    yield t

        def get(iterable, idx):
            return (v[idx] for v in iterable)
        gens = tee(gen(), len(iterables))
        return tuple(get(g, i) for i, g in enumerate(gens))

    def build(self):
        steps = self.data.steps
        counts = [0] * len(steps)
        frame_counters = 0
        for item in steps:
            if item.need_frame_no:
                frame_counters += 1
            for i in item.prev_idx:
                counts[i] += 1

        for i, n in enumerate(counts[:-1]):
            if n == 0 and steps[i].type != 'cull':
                print(f'Disabling unused step {i}: {steps[i].code}')
                # steps[i] = None

        program = []
        iterables = {}
        count_iterables = {}
        for i, (step, count) in enumerate(zip(steps, counts)):
            if step is not None and step.type != 'end':
                code = step.code
                for prev_idx in step.prev_idx:
                    prev_out = steps[prev_idx].out
                    its = iterables[prev_idx]
                    code = code.replace(prev_out, its.pop(0))
                if step.need_frame_no:
                    code = code.replace('frame_no', count_iterables[0].pop(0))
                program.append(code)
                if step.type == 'source':
                    if frame_counters > 0:
                        tees = [f'frame_no_{c + 1}' for c in range(frame_counters)]
                        count_iterables[i] = tees
                        outs = ', '.join(tees)
                        program.append(f'{outs} = ' + ', '.join(['range(len(src))'] * frame_counters))

                if step.type in {'source', 'func'}:
                    if count > 1:
                        tees = [f'{step.out}_{c + 1}' for c in range(count)]
                        iterables[i] = tees
                        outs = ', '.join(tees)
                        program.append(f'{outs} = tee({step.out}, {count})')
                    else:
                        iterables[i] = [step.out]
                elif step.type == 'cull':
                    its = ', '.join(sum(iterables.values(), []) + sum(count_iterables.values(), []))
                    program.append(f'{its} = cull({step.out}, {its})')
        ends = list(filter(lambda s: s.type == 'end', steps))
        end_vars = list(map(lambda e: steps[e.prev_idx[0]].out, ends))
        vs = ', '.join(end_vars)
        if len(end_vars) > 1:
            vs = f'zip({vs})'
        vsi = ', '.join(map(lambda ev: ev + '_item', end_vars))

        program.append(f'for {vsi} in {vs}:')
        program += list(map(lambda e: e.code, ends))
        print('Program is:')
        for i, line in enumerate(program):
            print(f'{i + 1:2}: {line}')
        program = '\n'.join(program)
        self.names = reduce(lambda r, s: r.update(s.names) or r, steps, {
            'tqdm': tqdm,
            'tee': tee,
            'grouper': grouper,
            'chain': chain,
            'cull': self._cull
        })
        self.program = compile(program, 'pipeline.py', 'exec')

    def __call__(self, source: Optional[Iterable[Any]] = None,
                 report_dir: Path = Path('reports'), data_slice: Optional[slice] = None):
        if self.program is None:
            self.build()
        for dumper in self.data.dumpers:
            dumper.reset(
                len(source) if data_slice is None else data_slice.stop - data_slice.start,
                0 if data_slice is None else data_slice.start
            )
        names = self.names
        names['src'] = source
        result = exec(self.program, names)
        for dumper in self.data.dumpers:
            dumper.finalize(report_dir)
        return result
