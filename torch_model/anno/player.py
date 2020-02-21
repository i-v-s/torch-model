from typing import Optional

from os import mkdir, listdir
from os.path import isdir, isfile, join

import numpy as np
import cv2
import xxhash

from torch_model import scale_with_padding


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


class Annotation:
    """Object of this class consists of annotation data and interface methods."""
    def __init__(self):
        return

    def clear(self):
        return

    def save(self, fn):
        raise NotImplementedError

    def on_key(self, key):
        return False

    def visualize(self, image, x, y):
        raise NotImplementedError

    def on_move(self, x, y):
        ...

    def on_left_down(self, x, y):
        ...

    def on_right_down(self, x, y):
        ...

    def on_left_up(self, x, y):
        ...

    def on_right_up(self, x, y):
        ...


class EggAnnotation(Annotation):
    def __init__(self):
        super().__init__()

    def get_handler(self):
        padding, rs = self.reduction
        scale = self.scale
        ss = scale * rs

        def rx(x):
            return (x // scale - padding) // rs

        def ry(x):
            return (x // scale - padding) // rs

        def circle(x, y, col):
            mask, mode = self.mask, self.mode - 1
            mm = mask[..., mode].copy()
            cv2.circle(mm, (rx(x), ry(y)), self.radius // ss, (col,), -1)
            mask[..., mode] = mm
            # mask[..., 3] = 255 - np.max(mask[..., :3], -1)

        def line(x, y, col):
            mask, mode = self.mask, self.mode - 1
            mm = mask[..., mode].copy()
            cv2.line(mm, self.start_pos, (rx(x), ry(y)), (col,), thickness=self.radius * 2 // ss)
            mask[..., mode] = mm
            # mask[..., 3] = 255 - np.max(mask[..., :3], -1)

        def handler(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                circle(x, y, 255)
                self.start_pos = rx(x), ry(y)
            elif event == cv2.EVENT_LBUTTONUP:
                circle(x, y, 255)
                line(x, y, 255)

            elif event == cv2.EVENT_RBUTTONDOWN:
                circle(x, y, 0)
                self.start_pos = rx(x), ry(y)
            elif event == cv2.EVENT_RBUTTONUP:
                circle(x, y, 0)
                line(x, y, 0)
            else:
                self.cursor = x, y
        return handler


class AnnoPlayer:
    def __init__(self, save_dir, annotation: Annotation, view_crop=None, scale=1, reduction=(0, 1), name='Image', radius=10, show_mask=True, roi_size=None):
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
        self.radius = radius
        self.roi_size = roi_size
        self.roi = None
        self.mode = 1
        self.show_mask = show_mask, 
        self.frame_name = None
        cv2.namedWindow(name, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(name, self.handler)
        self.mask: Optional[np.ndarray] = None
        self.frame = None
        self.cursor = None

    def handler(self, event, x, y, flags, param):
        if self.view_crop:
            ys, xs = self.view_crop
            x += xs.start
            y += ys.start

        if event == cv2.EVENT_LBUTTONDOWN:
            self.anno.on_left_down(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.anno.on_left_up(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.anno.on_right_down(x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.anno.on_right_up(x, y)
        else:
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
        index = 1
        while isfile(join(directory, 'images', f'{index:04d}.png')):
            index += 1
        fn = f'{index:04d}'
        #self.img_map[self.hash(frame)] = fn
        if self.anno.save(join(directory, 'masks', fn)):
            cv2.imwrite(join(directory, 'images', fn + '.png'), frame)
        #cv2.imwrite(join(directory, 'masks', fn + '.png'), mask[..., :3])
        self.last_mask = mask

    def zero_mask(self, frame):
        p, s = self.reduction
        size = tuple((m - 2 * p) // s for m in frame.shape[:2])
        mask = np.zeros(size + (4,), dtype=np.uint8)
        #mask[..., 3] = 255
        return mask

    def show(self, show_mask=False, writer=None):
        frame, mask, reduce = self.frame, self.mask, self.reduction
        if len(frame.shape) == 2:
            frame = np.expand_dims(frame, -1)
        if reduce != (0, 1):
            p, s = reduce
            mask = scale_with_padding(frame.shape, mask, s)
        alpha = 255 - np.max(mask[..., :3], -1, keepdims=True)
        image = (frame & mask[..., :3]) | (frame & alpha) if self.show_mask else frame
        if mask.shape[-1] == 1:
            image[:, :, 0] |= mask[:, :, 0]
        self.anno.visualize(image, 0, 0)
        if self.view_crop:
            image = image[self.view_crop]

        if writer:
            writer.write(mask)

        scale = self.scale
        if self.cursor:
            x, y = self.cursor
            cv2.circle(image, (x // scale, y // scale), self.radius, (0, 0, 255), -1)

        if scale != 1:
            h, w, c = frame.shape
            image = cv2.resize(image, (w * scale, h * scale))

        if self.frame_name:
            cv2.putText(image, self.frame_name, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (128, 255, 128))
        if show_mask:
            cv2.imshow('mask', mask[..., :3])
        if self.roi:
            p1, p2 = self.get_roi()
            cv2.rectangle(image, p1, p2, (0, 255, 128))
        cv2.imshow(self.name, image)

    def play(self, source, process, prepare=None, use_model=False, update=None, writer=None, delay=1):
        pause, read_one = True, True
        frame = None

        while True:
            if not pause or read_one:
                try:
                    frame = next(source)
                    if prepare is not None:
                        frame = prepare(frame)
                    self.frame_name = self.img_map.get(self.hash(frame), None)
                    # if self.frame_name:
                    #    pause = True
                except StopIteration:
                    break
                self.frame = frame
                result = process(frame) if use_model else self.zero_mask(frame)
                if type(result) is tuple:
                    self.mask, is_bad = result
                    if is_bad and self.frame_name is None:
                        pause = True
                else:
                    self.mask = result
                if self.mask.shape[-1] < 3:
                    self.mask = np.concatenate(
                        (self.mask, np.zeros(self.mask.shape[:2] + (3 - self.mask.shape[-1],), dtype=np.uint8)),
                        axis=-1
                    )

                read_one = False
                self.roi = None

            self.show(writer=None if pause else writer)
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
                pause = not pause
            elif k == ord('b'):
                source.back()
                read_one = True
            elif k == ord('m'):
                use_model = not use_model
                self.mask = process(frame) if use_model else self.zero_mask(frame)
                if type(self.mask) is tuple:
                    self.mask, is_bad = self.mask
                if self.mask.shape[-1] < 3:
                    self.mask = np.concatenate(
                        (self.mask, np.zeros(self.mask.shape[:2] + (3 - self.mask.shape[-1],), dtype=np.uint8)),
                        axis=-1
                    )

            elif k == ord('f'):
                source.forward()
                #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + 1)
                read_one = True
            elif k == ord('u'):
                if update:
                    for n in range(10):
                        try:
                            update()
                            break
                        except Exception as e:
                            print(f'Exception: {e}')
            elif k == ord('1'):
                self.mode = 1
            elif k == ord('2'):
                self.mode = 2
            elif k == ord('3'):
                self.mode = 3
            elif k == ord('r'):
                self.anno.clear()
                self.mask[:] = 0
                #self.mask[..., 3] = 255
            elif k == ord('='):
                self.radius += 1
            elif k == ord('-'):
                if self.radius > 1:
                    self.radius -= 1
            elif k == ord('l'):
                self.mask = self.last_mask
            elif k == ord('s'):
                if self.roi:
                    (x1, y1), (x2, y2) = self.get_roi()
                    self.save(frame[y1:y2, x1:x2], self.mask[y1:y2, x1:x2], f'{self.save_dir}-{self.roi_size}')
                else:
                    self.save(frame, self.mask, self.save_dir)
