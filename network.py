import argparse
import io
import math
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from streamlit_image_coordinates import streamlit_image_coordinates
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from itertools import combinations


TOOL_COLUMNS = [
    "code",
    "text_de",
    "text_sursilvan",
    "text_sutsilvan",
    "text_surmiran",
    "text_puter",
    "text_vallader"
]

OUT_PATH = Path("data/processed/draw-network")

POINT_RADIUS = 8
MARKER_PADDING = 2

COLORS = {
    "my_idiom":       (68, 119, 170),    # blue        #4477AA
    "romansh":        (238, 119, 51),    # orange      #EE7733
    "german":         (17, 119, 51),     # dark green  #117733
    "english":        (204, 51, 17),     # red         #CC3311
    "other_language": (221, 204, 119),   # sand        #DDCC77
}


LABELS_MONOLINGUAL = COLORS.keys()
LABELS_DUOLINGUAL = list(combinations(COLORS.keys(), 2))


SIDELENGTH = 600
MARGIN = 100
N_CIRCLES = 6
IMAGE_DICT = {
    "width": SIDELENGTH,
    "height": SIDELENGTH,
    "center": (SIDELENGTH/2, SIDELENGTH/2),
    "radii": [(SIDELENGTH-MARGIN)/(2*N_CIRCLES) * i for i in range(1, N_CIRCLES+1)],
    # sectors j = 1..6, boundaries at pi/6 * j, clockwise from the y axis
    "start_angle": 0.0,
    "clockwise": True,
    "line_color_circle": "black",
    "line_width_circle": 1.0,
    "line_color_sector": "grey",
    "line_width_sector": 1.0,}

st.set_page_config(layout="wide")

_t2p = TextToPath()

def _char_widths_px(text, fontsize, dpi):
    """Advance width of each character in pixels (incl. spaces)."""
    prop = FontProperties(size=fontsize)
    scale = dpi / 72  # points -> pixels
    return [_t2p.get_text_width_height_descent(c, prop, ismath=False)[0] * scale for c in text]


def curved_text(ax, text, angle, radius, center, fontsize=11, dpi=100,
                clockwise=True, align = "center", pad=3, letter_spacing=0.5, **kw):
    """Draw `text` along a circle, centred on `angle` (rad, clockwise from 12 o'clock).

    Axes must be in canvas pixel coords (y pointing down). Text in the lower
    half is flipped so it always reads left to right.
    """
    cx, cy = center
    sign = 1 if clockwise else -1
    widths = [w + letter_spacing for w in _char_widths_px(text, fontsize, dpi)]
    total = sum(widths) / radius                      # arc span in radians

    # position on screen, measured clockwise from top
    screen_a = (sign * angle) % (2 * math.pi)
    bottom = math.pi / 2 < screen_a < 3 * math.pi / 2
    step = -1 if bottom else 1                        # bottom half: run counter-clockwise
    if align == "start":
        a = screen_a + step * pad / radius
    else:
        a = screen_a - step * total / 2

    for ch, w in zip(text, widths):
        a_mid = a + step * (w / 2) / radius
        x = cx + radius * math.sin(a_mid)
        y = cy - radius * math.cos(a_mid)
        rot = -math.degrees(a_mid) + (180 if bottom else 0)
        ax.text(x, y, ch, rotation=rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=fontsize, **kw)
        a += step * w / radius

def _boundary_angles(cfg):
    """Angles of all sector boundaries, j = 0..n."""
    return [cfg["start_angle"] + j * cfg["sector_width"] for j in range(cfg["n_sectors"] + 1)]


def read_network_text(path: str | Path, idiom: str) -> pd.DataFrame:
    """Reads a CSV file with multilingual translations of the network task

    Args:
        path: Path to the file.
        idiom: The idiom to retrieve from the file
    Returns:
        The file as a frame, with all columns and rows preserved.

    Raises:
        ValueError: If any expected column is missing from the file.
    """

    frame: pd.DataFrame = pd.read_excel(path, dtype=str)
    expected = set(TOOL_COLUMNS)

    missing = expected - set(frame.columns)

    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")

    col = f"text_{idiom}"
    return frame[["code", col]].rename(columns={col: "text"})

# --- drawing -----------------------------------------------------------------
def draw_marker(draw: ImageDraw.ImageDraw, x: float, y: float,
                colours: tuple[tuple[int, int, int], ...],
                radius: float = POINT_RADIUS) -> None:
    box = [x - radius, y - radius, x + radius, y + radius]
    if len(colours) == 1:
        draw.ellipse(box, fill=colours[0])
    else:
        draw.pieslice(box, 90, 270, fill=colours[0])    # left half
        draw.pieslice(box, 270, 90, fill=colours[1])    # right half
    draw.ellipse(box, fill=None, outline=(0, 0, 0), width=2)

def draw_center_text(ax, text, cfg, fontsize=12, mask=True, **kw):
    cx, cy = cfg["center"]
    bbox = dict(boxstyle="circle,pad=0.3", fc="white", ec="none") if mask else None
    ax.text(cx, cy, text,
            ha="center", va="center",
            multialignment="center",
            fontsize=fontsize,
            zorder=10,
            bbox=bbox,
            **kw)

def draw_scale_label(ax, text, cfg, gap=14, fontsize=8):
    """Label curved along the outermost circle, centred on the start radius."""
    curved_text(ax, text, cfg["start_angle"], cfg["radii"][-1] - gap, cfg["center"],
                fontsize=fontsize, clockwise=cfg["clockwise"], color=cfg["line_color_sector"],
                zorder=10, align="start")

@st.cache_data
def marker_image(colours: tuple[tuple[int, int, int], ...],
                 supersample: int = 4) -> Image.Image:
    size = 2 * (POINT_RADIUS + MARKER_PADDING)
    img = Image.new("RGBA", (size * supersample,) * 2, (0, 0, 0, 0))
    draw_marker(ImageDraw.Draw(img),
                size * supersample / 2, size * supersample / 2,
                colours,
                radius=POINT_RADIUS * supersample)
    return img.resize((size, size), Image.Resampling.LANCZOS)

def ask_location(text: pd.DataFrame) -> None:

    title = text.loc[text["code"] == "describe_person", "text"].item()

    @st.dialog(title)
    def dialog(text: pd.DataFrame):
        index = st.session_state.pending
        point = st.session_state.points[index]





        closeness_codes = ["extremely_close", "very_close", "somewhat_close", "not_very_close"]
        closeness_labels = {c: text.loc[text["code"] == c, "text"].item() for c in closeness_codes}

        with st.form("location_form"):
            name = st.text_input(
                label=text.loc[text["code"] == "label_person", "text"].item() + " *",
                value=point["name"]            )
            location = st.text_input(
                label=text.loc[text["code"] == "location", "text"].item(),
                value=point["location"],
            )
            closeness = st.radio(
                label=text.loc[text["code"] == "emotional_closeness", "text"].item(),
                options=closeness_codes,
                format_func=lambda c: closeness_labels[c],
                index=closeness_codes.index(point["closeness"]) if point["closeness"] else None,
            )

            save_col, cancel_col = st.columns(2)
            confirmed = save_col.form_submit_button(
                text.loc[text["code"] == "save", "text"].item(), type="primary")
            cancelled = cancel_col.form_submit_button(text.loc[text["code"] == "cancel", "text"].item())

        st.caption(text.loc[text["code"] == "label_person_privacy", "text"].item())

        if confirmed:
            point["name"] = name.strip()
            point["location"] = location.strip()
            point["closeness"] = closeness
            st.session_state.pending = None
            st.rerun()
        if cancelled:
            st.session_state.points.pop(index)
            st.session_state.pending = None
            st.rerun()

    dialog(text)

def resolve_language_labels(text: pd.DataFrame) -> tuple[dict, dict]:
    lookup = text.set_index("code")["text"].to_dict()

    def resolve(codes: tuple[str, ...]) -> dict:
        return {"text": "  \n".join(lookup[c] for c in codes),
                "codes": codes,
                "color": tuple(COLORS[c] for c in codes)}

    mono = {c: resolve((c,)) for c in LABELS_MONOLINGUAL}
    duo = {"+".join(pair): resolve(pair) for pair in LABELS_DUOLINGUAL}
    return mono, duo

def draw_figure(cfg,
                text,
                labels=True):
    sectors = text.loc[text["code"].str.contains("_sector", regex=False, na=False), "code"].tolist()
    cfg["n_sectors"] = len(sectors)
    cfg["sector_width"] = 2*math.pi/cfg["n_sectors"]

    w, h = cfg["width"], cfg["height"]
    cx, cy = cfg["center"]
    r_start = cfg["radii"][0]
    r_end = cfg["radii"][-1]
    dpi = 100

    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi) # type: ignore[arg-type]
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_ylim(h, 0)  # y down -> same as canvas coords
    ax.set_aspect("equal")
    ax.axis("off")

    for r in cfg["radii"]:
        ax.add_patch(Circle((cx, cy), r, fill=False,
                            ec=cfg["line_color_circle"],
                            lw=cfg["line_width_circle"]))

    sign = 1 if cfg["clockwise"] else -1
    for a in _boundary_angles(cfg):
        x_start = cx + sign * r_start * math.sin(a)
        y_start = cy - r_start * math.cos(a)
        x_end = cx + sign * r_end * math.sin(a)
        y_end = cy - r_end * math.cos(a)
        ax.plot([x_start, x_end], [y_start, y_end],
                color=cfg["line_color_sector"],
                lw=cfg["line_width_sector"])

    if labels:

        for j, sector in enumerate(sectors):
            a = cfg["start_angle"] + (j + 0.5) * cfg["sector_width"]
            label = text.loc[text["code"] == sector, "text"].item()

            curved_text(ax, label, a, cfg["radii"][-1] + 16, cfg["center"],
                        fontsize=11, clockwise=cfg["clockwise"])
    ego = text.loc[text["code"] == "ego", "text"].item().upper()
    draw_center_text(ax, ego , cfg, fontsize=10)
    draw_scale_label(ax, text.loc[text["code"] == "frequency_label", "text"].item(), cfg)
    return fig


def make_background(cfg:dict, text:pd.DataFrame) -> Image.Image:
    fig = draw_figure(cfg, text)          # the drawing part of draw_background, minus savefig
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_canvas(base_img: Image.Image,
                  points: list[dict], categories:dict) -> Image.Image:
    canvas_img = base_img.copy()
    draw = ImageDraw.Draw(canvas_img)
    font = ImageFont.load_default(size=13)
    for pt in points:
        draw_marker(draw, pt['x'], pt['y'], categories[pt['category']]['color'])
        if pt['name']:
            draw.text((pt['x'], pt['y'] + POINT_RADIUS + 3), pt['name'],
                      fill=(0, 0, 0), font=font,
                      anchor="ma", stroke_width=3, stroke_fill=(255, 255, 255))
    return canvas_img


def to_records(points: list[dict], width: int, height: int) -> pd.DataFrame:
    records = pd.DataFrame(points)
    records["x_rel"] = records["x"] / width
    records["y_rel"] = 1 - records["y"] / height
    return records


def write(records: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_path, mode="a", header=not out_path.exists(), index=False)


# --- state -------------------------------------------------------------------
def init_state(categories) -> None:
    st.session_state.setdefault("points", [])
    st.session_state.setdefault("last_click", None)
    st.session_state.setdefault("category", next(iter(categories)))
    st.session_state.setdefault("submitted", False)
    st.session_state.setdefault("pending", None)


def add_point(coords: dict, base_width: int, category: str) -> bool:
    current_click = (coords["x"], coords["y"])
    if current_click == st.session_state.last_click:
        return False
    st.session_state.last_click = current_click
    scale = base_width / coords.get("width", base_width)
    st.session_state.points.append({
        'x': round(coords["x"] * scale),
        'y': round(coords["y"] * scale),
        'category': category,
        'location': "",
        'name':"",
        'closeness':"",
    })
    st.session_state.pending = len(st.session_state.points) - 1
    return True


# --- rendering ---------------------------------------------------------------
def render_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] { max-width: 1300px; }
        h1 { text-align: center; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_category_buttons(categories: dict) -> None:
    for key, value in categories.items():
        dot, label = st.columns([1, 6], vertical_alignment="center")
        dot.image(marker_image(value['color']))
        active = key == st.session_state.category
        if label.button(value['text'], key=f"cat_{key}",
                        type="primary" if active else "secondary",
                        width='stretch'):
            st.session_state.category = key
            st.rerun()

def render_legend(text: pd.DataFrame,
                  mono_categories: dict,
                  duo_categories: dict) -> str:
    st.markdown(text.loc[text["code"] == "language_header", "text"].item())

    mono_tab, duo_tab = st.tabs([
        text.loc[text["code"] == "monolingual_interaction", "text"].item(),
        text.loc[text["code"] == "duolingual_interaction", "text"].item(),
    ])

    with mono_tab:
        with st.container(height=360, border=False):
            render_category_buttons(mono_categories)

    with duo_tab:
        with st.container(height=360, border=False):
            render_category_buttons(duo_categories)

    return st.session_state.category

# add dynamic language
def render_controls(base_img: Image.Image, out_path: Path, text:pd.DataFrame) -> None:
    col1, col2, col3 = st.columns(3)
    remove_point = text.loc[text["code"] == "remove_point", "text"].item()
    remove_all_points = text.loc[text["code"] == "remove_all_points", "text"].item()
    save = text.loc[text["code"] == "save", "text"].item()
    saved = text.loc[text["code"] == "saved", "text"].item()
    not_saved = text.loc[text["code"] == "not_saved", "text"].item()

    with col1:
        if st.button(remove_point) and st.session_state.points:
            st.session_state.points.pop()
            st.rerun()

    with col2:
        if st.button(remove_all_points):
            st.session_state.points = []
            st.rerun()

    with col3:
        if st.button(save, type="primary", disabled=not st.session_state.points):
            records = to_records(st.session_state.points, base_img.width, base_img.height)
            try:
                write(records, out_path)
            except Exception as exc:
                st.error(f"{not_saved}{exc}")
            else:
                st.session_state.submitted = True
                st.toast(saved, icon="✅")
                st.rerun()


# --- entry point -------------------------------------------------------------
def run(
        network_text_path: Path,
        idiom: str,
) -> None:
    render_styles()

    # Get idiom of a participant with a specific ID
    # Bypass this logic in case of unexpected errors

    text = read_network_text(network_text_path, idiom)

    st.title(text.loc[text["code"] == "header", "text"].item())

    # Do not show participant ID but information about the participant (location, gender, age group)
    participant = f"{text.loc[text["code"] == "participant", "text"].item()}: {idiom}"
    st.caption(participant)

    mono_categories, duo_categories = resolve_language_labels(text)
    categories = mono_categories | duo_categories
    init_state(categories)

    base_img = make_background(IMAGE_DICT, text)
    canvas_img = render_canvas(base_img, st.session_state.points, categories)

    img_col, legend_col = st.columns([2, 1], vertical_alignment="top")
    with legend_col:
        category = render_legend(text, mono_categories, duo_categories)
    with img_col:
        coords = streamlit_image_coordinates(canvas_img, key="annotator")

    if coords is not None and add_point(coords, base_img.width, category):
        st.rerun()

    if st.session_state.pending is not None:
        ask_location(text)

    # Change to participant_id
    out_path = OUT_PATH / f"{idiom}_response.csv"
    render_controls(base_img, out_path, text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idiom", required=True)
    args = parser.parse_args()

    run(idiom=args.idiom,
        network_text_path=Path("network_text.xlsx")
        )


if __name__ == "__main__":
    main()
