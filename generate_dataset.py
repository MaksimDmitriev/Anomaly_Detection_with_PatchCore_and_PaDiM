"""Generate a toy concrete-surface dataset for anomaly detection learning.

All images are SYNTHETIC. This is not inspection-grade data.
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).parent / 'data'
SIZE = 192
RNG = np.random.default_rng(42)


def normal_surface():
    # Smooth, roughly concrete-like texture with a faint vertical panel seam.
    base = float(RNG.uniform(147, 173))
    noise = RNG.normal(0, 8, (SIZE, SIZE))
    gradual_light = np.linspace(-7, 7, SIZE)[None, :]
    mono = np.clip(base + noise + gradual_light, 0, 255).astype(np.uint8)
    image = Image.fromarray(mono, mode='L').convert('RGB')
    draw = ImageDraw.Draw(image)
    seam_x = int(RNG.integers(36, 48))
    draw.line([(seam_x, 0), (seam_x, SIZE)], fill=(112, 112, 112), width=2)
    draw.line([(seam_x + 3, 0), (seam_x + 3, SIZE)], fill=(186, 186, 186), width=2)
    return image


def add_defect(image, kind):
    draw = ImageDraw.Draw(image)
    if kind == 'crack':
        x, y = int(RNG.integers(82, 118)), int(RNG.integers(35, 60))
        points = [(x, y)]
        for _ in range(9):
            x += int(RNG.integers(-9, 11))
            y += int(RNG.integers(8, 13))
            points.append((x, y))
        draw.line(points, fill=(31, 30, 29), width=4, joint='curve')
        branch = [points[4], (points[4][0]+24, points[4][1]+8),
                  (points[4][0]+39, points[4][1]+23)]
        draw.line(branch, fill=(46, 38, 34), width=2)
    elif kind == 'corrosion':
        cx, cy = int(RNG.integers(87, 132)), int(RNG.integers(57, 129))
        # Irregular blotches suggesting a rust-like spot.
        for _ in range(42):
            dx, dy = RNG.normal(0, 11, 2)
            rad = int(RNG.integers(2, 7))
            x, y = cx + int(dx), cy + int(dy)
            tone = int(RNG.integers(58, 105))
            draw.ellipse((x-rad, y-rad, x+rad, y+rad),
                         fill=(min(255,tone+57), tone, max(0,tone-34)))
    else:
        raise ValueError(kind)
    return image


def save(split, subtype, n):
    folder = ROOT / split / subtype
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = normal_surface()
        if subtype == 'defect':
            img = add_defect(img, 'crack' if i % 2 == 0 else 'corrosion')
        img.save(folder / f'{split}_{subtype}_{i:03d}.png')


def make_preview():
    from PIL import ImageOps, ImageFont
    source = [ROOT/'train/good/train_good_000.png',
              ROOT/'test/defect/test_defect_000.png',
              ROOT/'test/defect/test_defect_001.png']
    labels = ['Normal', 'Defect: crack', 'Defect: corrosion']
    canvas = Image.new('RGB', (3 * SIZE, SIZE + 26), (250,250,250))
    draw = ImageDraw.Draw(canvas)
    for j,(path,label) in enumerate(zip(source,labels)):
        canvas.paste(Image.open(path), (j*SIZE,26))
        draw.text((j*SIZE+8,8), label, fill=(30,30,30))
    canvas.save(Path(__file__).parent / 'sample_preview.png')


if __name__ == '__main__':
    save('train','good',60)
    save('test','good',16)
    save('test','defect',16)
    save('demo','good',2)
    save('demo','defect',2)
    make_preview()
    print('Created 60 training normal images, 16 normal test + 16 defective test images.')
    print('Generated sample_preview.png; the demo images are outside the evaluation set.')
