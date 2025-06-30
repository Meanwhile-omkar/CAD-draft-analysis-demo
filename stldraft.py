import os
import tempfile
import logging
import numpy as np
import streamlit as st
import random

try:
    import pyvista as pv
except ImportError as e:
    st.error("PyVista is required. Please install pyvista.")
    raise e

# Logging setup
tq_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

st.set_page_config(page_title="STL Draft Angle Analysis", layout="wide")
st.title("📐 Draft Analysis")

@st.cache_data(show_spinner="Loading mesh…")
def load_mesh_picklable(path):
    try:
        tq_logger.info(f"Reading mesh {path}")
        mesh = pv.read(path)
        tq_logger.info(f"Loaded: {mesh.n_points} pts, {mesh.n_cells} cells.")
        return mesh
    except Exception as e:
        tq_logger.error(f"Mesh read failed: {e}")
        st.error(f"Mesh read failed: {e}")
        return None

@st.cache_data(show_spinner="Extracting normals/centers…")
def extract_normals_and_centers(_mesh):
    try:
        normals = _mesh.face_normals
        centers = _mesh.cell_centers().points
        tq_logger.info(f"Normals/centers: {len(normals)} items.")
        return normals, centers
    except Exception as e:
        tq_logger.error(f"Extract failed: {e}")
        st.error(f"Extract failed: {e}")
        return None, None

@st.cache_resource(show_spinner="Decimating mesh…")
def decimate_mesh(mesh, reduction=0.5):
    try:
        tq_logger.info(f"Decimating: reduction={reduction}")
        dec = mesh.decimate_pro(n_reduction=reduction)
        tq_logger.info(f"Decimated: {dec.n_points} pts, {dec.n_cells} cells.")
        return dec
    except Exception as e:
        tq_logger.warning(f"Decimate error: {e}")
        return mesh

# ----- Step 1: Upload -----
uploaded = st.file_uploader("Upload STL file", type=["stl"])
if not uploaded:
    st.info("Awaiting STL file...")
    st.stop()

suffix = uploaded.name.split('.')[-1].lower()
tmpf = tempfile.NamedTemporaryFile(suffix=f'.{suffix}', delete=False)
tmpf.write(uploaded.read())
tmpf.close()
path = tmpf.name
tq_logger.info(f"Saved upload to {path}")

# ----- Step 2: Settings -----
st.subheader("Settings")
threshold_angle = st.number_input("Draft angle threshold (degrees)", min_value=0.0, max_value=45.0, value=2.0, step=0.5)
max_snaps = st.number_input("Max screenshots of draft issues", min_value=1, max_value=20, value=5, step=1)
run_analysis = st.button("Run Analysis")

# Fixed main tooling axis (Y direction)
main_tooling_dir = np.array([0, 1, 0], dtype=float)
main_tooling_dir = main_tooling_dir / np.linalg.norm(main_tooling_dir)  # Ensure unit vector

if not run_analysis:
    st.info("Set your preferences and click 'Run Analysis'.")
    st.stop()

# ----- Step 3: Analysis -----
mesh = load_mesh_picklable(path)
if mesh is None:
    st.stop()

if mesh.n_cells > 1_000_000:
    mesh = decimate_mesh(mesh, 0.7)

normals, centers = extract_normals_and_centers(mesh)
if normals is None or centers is None:
    st.stop()

# Mesh-based draft angle analysis
st.subheader("2️⃣ Draft-angle analysis (mesh-based)…")
with st.spinner("Draft-angle analysis (mesh-based)…"):
    draft_issues = []
    for i, normal in enumerate(normals):
        angle = np.arccos(np.clip(np.dot(normal, main_tooling_dir), -1.0, 1.0)) * 180 / np.pi
        if angle < threshold_angle:
            pt = centers[i]
            draft_issues.append({
                "point": pt,
                "angle": angle,
                "index": i
            })
    st.write(f"• Checked {len(normals)} mesh faces")
    st.success(f"✔ {len(draft_issues)} faces found with draft < {threshold_angle:.1f}°")

issue_points = None
if draft_issues:
    issue_points = np.stack([issue["point"] for issue in draft_issues])
    # Number for table (1-based, as seen by user)
    for n, issue in enumerate(draft_issues, 1):
        issue["number"] = n

# --- 3D Snapshot with main tooling axis arrows for issues ---
st.subheader("🧊 3D Snapshot with Main Tooling Axis and Issue Vectors")
bounds = np.array(mesh.bounds)
center = (bounds[::2] + bounds[1::2]) / 2
extent = np.linalg.norm(bounds[1::2] - bounds[::2])
factor = extent * 0.05 if extent > 0 else 1.0
length = extent * 1.5

arrow_start = center - main_tooling_dir * (length/2)
arrow_end = center + main_tooling_dir * (length/2)
main_arrow = pv.Arrow(
    arrow_start,
    main_tooling_dir,
    tip_length=0.3,
    tip_radius=0.01,
    shaft_radius=0.001,
    scale=length
)

# Normals glyphs
point_data = pv.PolyData(centers)
point_data["normals"] = normals
glyphs = point_data.glyph(orient="normals", scale=False, factor=factor)

p = pv.Plotter(off_screen=True)
p.set_background("white")
p.add_mesh(mesh, show_edges=True)
p.add_mesh(glyphs, color="red")

if issue_points is not None:
    for pt, issue in zip(issue_points, draft_issues):
        issue_arrow = pv.Arrow(pt - main_tooling_dir * factor, main_tooling_dir, tip_length=0.3, tip_radius=0.04, shaft_radius=0.02, scale=factor*2)
        p.add_mesh(issue_arrow, color="blue")

p.add_mesh(main_arrow, color="yellow", name="tooling_axis")
p.add_point_labels([arrow_end], ["Main Tooling Direction"], point_color="yellow", font_size=24, text_color="black")

st.image(p.screenshot(), caption="3D Model with Main Tooling Axis and Issue Arrows")

# --- Issues table with coordinates and numbering ---
if draft_issues:
    st.subheader("📋 Issues Table (Coordinates)")
    coords = [np.round(issue["point"], 3) for issue in draft_issues]
    angles = [np.round(issue["angle"], 2) for issue in draft_issues]
    numbers = [issue["number"] for issue in draft_issues]
    table_data = {
        "No.": numbers,
        "X": [c[0] for c in coords],
        "Y": [c[1] for c in coords],
        "Z": [c[2] for c in coords],
        "Draft Angle (deg)": angles
    }
    st.dataframe(table_data)

# --- Random snapshot selection and display ---
if draft_issues:
    st.subheader("⚠️ Selected Draft Issues")
    random.seed(42)  # for reproducibility; remove or modify for true randomness each run
    n_select = min(max_snaps, len(draft_issues))
    selected_issues = random.sample(draft_issues, n_select)
    # sort selected by their table number
    selected_issues = sorted(selected_issues, key=lambda x: x["number"])
    for issue in selected_issues:
        pt = tuple(np.round(issue['point'], 2))
        st.write(f"{issue['number']}. Angle {issue['angle']:.2f}° at {pt}")

# --- Snapshots of Selected Issue Points ---
if draft_issues and selected_issues:
    st.subheader("🖼️ Snapshots of Selected Issues")
    images = []
    for issue in selected_issues:
        pt = issue["point"]
        snap = pv.Plotter(off_screen=True)
        snap.set_background("white")
        snap.add_mesh(mesh, show_edges=True)
        issue_arrow = pv.Arrow(pt - main_tooling_dir * factor, main_tooling_dir, tip_length=0.3, tip_radius=0.04, shaft_radius=0.02, scale=factor*2)
        snap.add_mesh(issue_arrow, color="blue")
        snap.add_mesh(main_arrow, color="yellow")
        snap.add_point_labels([arrow_end], ["Main Tooling Direction"], point_color="yellow", font_size=24, text_color="black")
        snap.camera_position = [(pt + np.array([extent/2, extent/2, extent/2])).tolist(), pt.tolist(), [0, 0, 1]]
        img = snap.screenshot()
        images.append((issue["number"], img))
    for number, img in images:
        st.image(img, caption=f"Issue {number}")

try:
    os.unlink(path)
    tq_logger.info(f"Removed {path}")
except Exception:
    pass