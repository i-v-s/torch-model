from typing import Optional
from os import mkdir, listdir
from os.path import isdir, isfile, join, splitext

import numpy as np
import cv2
import xxhash

from torch_model import scale_with_padding
from .annotation import Annotation
from .transform import AnnoTransform


class AnnoPlayer:
    def __init__(self, save_dir, annotation: Annotation, transform: AnnoTransform = None,
                 scale=1, reduction=(0, 1), name='Image', show_mask=True, roi_size=None):
        self.anno: Annotation = annotation
        annotation.set_player(self)
        self.img_map = {}
        self.xxhash = xxhash.xxh64()
        self.save_dir = save_dir
        self.transform = transform
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
        self.source = None

    def handler(self, event, x, y, flags, param):
        self.cursor = self.transform.invert_point(x, y) if self.transform else (x, y)
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
                if fn.endswith('.png') and isfile(join(images, fn))
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

        if self.transform:
            image = self.transform(image)

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
        self.source = source
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
