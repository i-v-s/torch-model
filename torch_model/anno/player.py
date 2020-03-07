from typing import Optional, Tuple

from os import mkdir, listdir
from os.path import isdir, isfile, join, splitext

import numpy as np
import cv2
import xxhash

from torch_model import scale_with_padding, load_model, process_image

class PNGSource:
    """Simple OpenCV source"""
    def __init__(self, directory, num_sort=True):
        self.directory = directory
        files = [fn for fn in listdir(directory) if fn.endswith('.png') and isfile(join(directory, fn))]
        if num_sort:
            self.files = sorted(files, key=lambda fn: int(fn[:-4]))
        else:
            self.files = sorted(files)
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.files):
            raise StopIteration
        frame = cv2.imread(join(self.directory, self.files[self.idx]))
        self.idx += 1
        return frame

    def forward(self, frames=200):
        self.idx += frames

    def back(self, frames=1):
        self.idx -= frames + 1

    def frame_no(self) -> int:
        return self.idx

    def __len__(self) -> int:
        return len(self.files)


class OCVSource:
    """Simple OpenCV source"""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            flag, frame = self.cap.read()
            if not flag:
                raise StopIteration
            return frame

    def forward(self, frames=200):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + frames)

    def back(self, frames=1):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) - frames - 1)

    def frame_no(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

    def __len__(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


class Annotation:
    """Object of this class consists of annotation data and interface methods."""
    def __init__(self, model_name: Optional[str] = None, device=None):
        self.model_name = model_name
        self.device = device
        self.model = None
        if model_name is not None:
            self.update_model()

    def update_model(self):
        self.model = load_model(self.model_name, device=self.device)

    def process(self, frame: np.ndarray):
        raise NotImplementedError

    def clear(self):
        return

    def save(self, fn: str) -> bool:
        raise NotImplementedError

    def load(self, fn: str):
        ...

    def on_key(self, key) -> bool:
        return False

    def visualize(self, image: np.ndarray, cursor: Optional[Tuple[int, int]], frame_no: int):
        raise NotImplementedError

    def on_move(self, x: int, y: int):
        ...

    def on_left_down(self, x: int, y: int):
        ...

    def on_right_down(self, x: int, y: int):
        ...

    def on_left_up(self, x: int, y: int):
        ...

    def on_right_up(self, x: int, y: int):
        ...


class SegAnnotation(Annotation):
    def __init__(self, channels=3, model_name: Optional[str] = None, device=None, radius=10, opacity: float = 0.7):
        super().__init__(model_name, device)
        if type(channels) is int:
            channels = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255))[:channels]
        self.channels = np.array(channels, dtype=np.uint8)
        self.opacity = opacity
        self.active_channel = 0
        self.mask = None
        self.keys = {ord('1') + c: c for c in range(len(channels))}
        self.start_pos = None
        self.radius = radius

    def on_key(self, key):
        c = self.keys.get(key, None)
        if c is not None:
            self.active_channel = c
        elif key == ord('='):
            self.radius += 1
        elif key == ord('-'):
            if self.radius > 1:
                self.radius -= 1
        else:
            return False
        return True

    def circle(self, x, y, col):
        mask = self.mask[:, :, self.active_channel]
        mm = mask.copy()
        cv2.circle(mm, (x, y), self.radius, (col,), -1)
        mask[:] = mm

    def line(self, x, y, col):
        mask = self.mask[:, :, self.active_channel]
        mm = mask.copy()
        cv2.line(mm, self.start_pos, (x, y), (col,), thickness=self.radius * 2)
        mask[:] = mm

    def on_left_down(self, x, y):
        self.circle(x, y, 255)
        self.start_pos = x, y

    def on_left_up(self, x: int, y: int):
        self.circle(x, y, 255)
        self.line(x, y, 255)

    def on_right_down(self, x: int, y: int):
        self.circle(x, y, 0)
        self.start_pos = x, y

    def on_right_up(self, x: int, y: int):
        self.circle(x, y, 0)
        self.line(x, y, 0)

    def visualize(self, image: np.ndarray, cursor):
        if self.mask is None:
            shape = image.shape[:2]
            self.mask = np.zeros(shape + (len(self.channels),), dtype=image.dtype)
        mask = (np.expand_dims(self.mask, 3).astype(np.float32) * self.channels).sum(2) * (self.opacity / 255)
        image[:] = np.clip(image + mask, 0, 255).astype(np.uint8)
        if cursor is not None:
            x, y = cursor
            cv2.circle(image, (x, y), self.radius, (255, 255, 255), -1)

    def process(self, frame: np.ndarray):
        self.mask = process_image(frame, self.model)

    def save(self, fn: str):
        if len(self.channels) > 3:
            raise NotImplementedError
        mask = self.mask
        # if len(mask.shape) == 3 and mask.shape[2] in ['2']:
        cv2.imwrite(fn + '.png', mask)
        return True

    def clear(self):
        self.mask[:] = 0


class AnnoPlayer:
    def __init__(self, save_dir, annotation: Annotation, view_crop=None, scale=1, reduction=(0, 1), name='Image', show_mask=True, roi_size=None):
        self.anno: Annotation = annotation
        self.img_map = {}
        self.xxhash = xxhash.xxh64()
        self.save_dir = save_dir
        self.view_crop = view_crop
        if roi_size is not None:
            self.create_dirs(f'{save_dir}-{roi_size}')
        self.create_dirs(save_dir)
        self.scale = scale
        self.reduction = reduction
        self.name = name
        self.roi_size = roi_size
        self.roi = None
        self.mode = 1
        self.hide = False
        self.frame_name = None
        cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(name, self.handler)
        self.mask: Optional[np.ndarray] = None
        self.frame = None
        self.cursor = None
        self.pause = True
        self.use_model = False

    def handler(self, event, x, y, flags, param):
        if self.view_crop:
            ys, xs = self.view_crop
            x += xs.start
            y += ys.start
        self.cursor = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            self.anno.on_left_down(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.anno.on_left_up(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.anno.on_right_down(x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.anno.on_right_up(x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            self.anno.on_move(x, y)

    def get_roi(self):
        if not self.roi:
            return None
        x, y = self.roi
        rs2 = self.roi_size // 2
        x = max(rs2, min(self.frame.shape[1] - rs2, x))
        y = max(rs2, min(self.frame.shape[0] - rs2, y))
        return (x - rs2, y - rs2), (x + rs2, y + rs2)

    def hash(self, x):
        xh = self.xxhash
        xh.reset()
        xh.update(x)
        return xh.hexdigest()

    def create_dirs(self, save_dir):
        if not isdir(save_dir):
            mkdir(save_dir)
        images, masks = join(save_dir, 'images'), join(save_dir, 'masks')
        if isdir(images):
            self.img_map = {
                self.hash(cv2.imread(join(images, fn))): fn
                for fn in listdir(images)
                if isfile(join(images, fn))
            }
        else:
            mkdir(images)
        if not isdir(masks):
            mkdir(masks)

    def save(self, frame, mask, directory):
        if self.frame_name is not None:
            fn = splitext(self.frame_name)[0]
        else:
            index = 1
            while isfile(join(directory, 'images', f'{index:04d}.png')):
                index += 1
            fn = f'{index:04d}'
        if self.anno.save(join(directory, 'masks', fn)):
            cv2.imwrite(join(directory, 'images', fn + '.png'), frame)
            self.img_map[self.hash(frame)] = fn
        #cv2.imwrite(join(directory, 'masks', fn + '.png'), mask[..., :3])
        self.last_mask = mask

    def zero_mask(self, frame):
        p, s = self.reduction
        size = tuple((m - 2 * p) // s for m in frame.shape[:2])
        mask = np.zeros(size + (4,), dtype=np.uint8)
        #mask[..., 3] = 255
        return mask

    def show(self, frame_no: int, writer=None):
        frame, mask, reduce = self.frame, self.mask, self.reduction
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, -1)
        if reduce != (0, 1):
            p, s = reduce
            mask = scale_with_padding(frame.shape, mask, s)
        alpha = 255 - np.max(mask[..., :3], -1, keepdims=True)
        image = frame.copy()
        if mask.shape[-1] == 1:
            image[:, :, 0] |= mask[:, :, 0]
        if not self.hide:
            self.anno.visualize(image, self.cursor, frame_no)

        if self.view_crop:
            image = image[self.view_crop]

        if writer:
            writer.write(mask)

        scale = self.scale

        if scale != 1:
            h, w, c = frame.shape
            image = cv2.resize(image, (w * scale, h * scale))

        if self.frame_name:
            cv2.putText(image, self.frame_name, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 255, 128))
        if self.hide:
            cv2.putText(image, 'H', (image.shape[1] - 40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        if self.pause:
            cv2.putText(image, 'P', (image.shape[1] - 70, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
        if self.use_model:
            cv2.putText(image, 'M', (image.shape[1] - 100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))

        if self.roi:
            p1, p2 = self.get_roi()
            cv2.rectangle(image, p1, p2, (0, 255, 128))
        cv2.imshow(self.name, image)

    def play(self, source, prepare=None, use_model=False, update=None, writer=None, delay=1):
        self.use_model = use_model
        self.pause, read_one = True, True
        frame, frame_no = None, None

        while True:
            if not self.pause or read_one:
                try:
                    frame_no = source.frame_no()
                    frame = next(source)
                    if prepare is not None:
                        frame = prepare(frame)
                    self.frame_name = self.img_map.get(self.hash(frame), None)
                    if self.frame_name is not None:
                        self.anno.load(join(self.save_dir, 'masks', splitext(self.frame_name)[0]))
                    # if self.frame_name:
                    #    pause = True
                except StopIteration:
                    break
                self.frame = frame
                if self.use_model:
                    self.anno.process(frame)
                result = self.zero_mask(frame) # self.anno.process(frame) if use_model else
                if type(result) is tuple:
                    self.mask, is_bad = result
                    if is_bad and self.frame_name is None:
                        self.pause = True
                else:
                    self.mask = result
                if self.mask.shape[-1] < 3:
                    self.mask = np.concatenate(
                        (self.mask, np.zeros(self.mask.shape[:2] + (3 - self.mask.shape[-1],), dtype=np.uint8)),
                        axis=-1
                    )

                read_one = False
                self.roi = None

            self.show(frame_no, writer=None if self.pause else writer)
            k = cv2.waitKey(delay)
            if k == -1 or self.anno.on_key(k):
                continue
            elif k == ord('Q'): # Left
                self.mask = np.concatenate((self.mask[:, 1:], self.mask[:, :1]))
            elif k == ord('S'): # Right
                self.mask = np.concatenate((self.mask[:, -1:], self.mask[:, :-1]))
            elif k == ord('q'):
                break
            elif k == ord('i'):
                self.roi = self.cursor
            elif k == ord('p'):
                self.pause = not self.pause
            elif k == ord('b'):
                source.back()
                read_one = True
            elif k == ord('m'):
                self.use_model = not self.use_model
                self.anno.process(frame) if self.use_model else self.zero_mask(frame)
                if type(self.mask) is tuple:
                    self.mask, is_bad = self.mask
                if self.mask.shape[-1] < 3:
                    self.mask = np.concatenate(
                        (self.mask, np.zeros(self.mask.shape[:2] + (3 - self.mask.shape[-1],), dtype=np.uint8)),
                        axis=-1
                    )
            elif k == ord('n'):
                read_one = True
            elif k == ord('h'):
                self.hide = not self.hide
            elif k == ord('f'):
                source.forward()
                read_one = True
            elif k == ord('u'):
                if update:
                    for n in range(10):
                        try:
                            update()
                            break
                        except Exception as e:
                            print(f'Exception: {e}')
            elif k == ord('r'):
                self.anno.clear()
            elif k == ord('l'):
                self.mask = self.last_mask
            elif k == ord('s'):
                if self.roi:
                    (x1, y1), (x2, y2) = self.get_roi()
                    self.save(frame[y1:y2, x1:x2], self.mask[y1:y2, x1:x2], f'{self.save_dir}-{self.roi_size}')
                else:
                    self.save(frame, self.mask, self.save_dir)
