"""Small dependency-free plotting helpers for diagnostic reports (Pillow)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def canvas(title: str, subtitle: str, width: int = 1600, height: int = 1000) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "#f7f9fc")
    draw = ImageDraw.Draw(image)
    draw.text((48, 25), title, font=font(28), fill="#17283d")
    draw.text((48, 65), subtitle, font=font(17), fill="#54677c")
    return image, draw


def line_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    series: Sequence[tuple[str, Sequence[tuple[float, float]], str]],
    *,
    x_label: str,
    y_label: str,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    diagonal: bool = False,
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=12, fill="white", outline="#dce4ee", width=2)
    draw.text((x0 + 18, y0 + 12), title, font=font(19), fill="#192c41")
    left, top, right, bottom = x0 + 85, y0 + 52, x1 - 28, y1 - 65
    xmin, xmax = x_range; ymin, ymax = y_range
    if xmax <= xmin or ymax <= ymin:
        return
    def xy(x: float, y: float) -> tuple[float, float]:
        return (left + (x - xmin) / (xmax - xmin) * (right - left),
                bottom - (y - ymin) / (ymax - ymin) * (bottom - top))
    for i in range(6):
        x = left + (right - left) * i / 5
        y = bottom - (bottom - top) * i / 5
        draw.line((x, top, x, bottom), fill="#e8eef4", width=1)
        draw.line((left, y, right, y), fill="#e8eef4", width=1)
        draw.text((x - 14, bottom + 8), f"{xmin + (xmax-xmin)*i/5:.2g}", font=font(13), fill="#617386")
        draw.text((left - 58, y - 7), f"{ymin + (ymax-ymin)*i/5:.2g}", font=font(13), fill="#617386")
    draw.line((left, bottom, right, bottom), fill="#344c63", width=2)
    draw.line((left, top, left, bottom), fill="#344c63", width=2)
    draw.text((left + (right-left)//3, bottom + 31), x_label, font=font(14), fill="#344c63")
    draw.text((x0 + 8, top - 19), y_label, font=font(13), fill="#344c63")
    if diagonal and xmin == ymin and xmax == ymax:
        draw.line((*xy(xmin, ymin), *xy(xmax, ymax)), fill="#a5b5c6", width=2)
    legend_y = top + 4
    for label, points, color in series:
        valid = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y) and xmin <= x <= xmax and ymin <= y <= ymax]
        coords = [xy(x, y) for x, y in valid]
        if len(coords) > 1:
            draw.line(coords, fill=color, width=3)
        for x, y in coords:
            draw.ellipse((x-4, y-4, x+4, y+4), fill=color)
        if label:
            draw.line((right - 145, legend_y + 8, right - 127, legend_y + 8), fill=color, width=3)
            draw.text((right - 122, legend_y), label, font=font(12), fill="#344c63")
            legend_y += 18


def save(image: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def scatter_panel(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    title: str,
    points: Iterable[tuple[float, float]],
    *, x_label: str, y_label: str,
    x_range: tuple[float, float], y_range: tuple[float, float], color: str = "#227c9d",
) -> None:
    line_panel(draw, rect, title, [], x_label=x_label, y_label=y_label,
               x_range=x_range, y_range=y_range)
    x0, y0, x1, y1 = rect
    left, top, right, bottom = x0 + 85, y0 + 52, x1 - 28, y1 - 65
    xmin, xmax = x_range; ymin, ymax = y_range
    for x, y in points:
        if not (math.isfinite(x) and math.isfinite(y)) or not (xmin <= x <= xmax and ymin <= y <= ymax):
            continue
        px = left + (x - xmin) / (xmax - xmin) * (right - left)
        py = bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
        draw.ellipse((px-3, py-3, px+3, py+3), fill=color)
