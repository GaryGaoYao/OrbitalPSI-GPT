import vtk
import os
import math
from tqdm import tqdm

# ==== 输入输出路径 ====
input_dir = "D:/Codes/Skull_Landmarks_TL/skull_models"   # STL 文件所在文件夹
output_dir = "D:/Codes/Skull_Landmarks_TL/skull_models_2d_pics_improved"  # 截图保存的文件夹
os.makedirs(output_dir, exist_ok=True)

# ==== 遍历所有 STL 文件 ====
stl_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".stl")]
if not stl_files:
    print("未在输入目录中找到 .stl 文件。")
    raise SystemExit

# ==== 渲染器、窗口、截图工具（复用） ====
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(800, 800)           # 正方形窗口
render_window.SetMultiSamples(8)
render_window.SetOffScreenRendering(1)    # 离屏渲染

camera = renderer.GetActiveCamera()
w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window)
png_writer = vtk.vtkPNGWriter()

# ==== 扰动范围（很小，保证完整显示） ====
YAW_RANGE_DEG   = 3     # ±3°
PITCH_RANGE_DEG = 2     # ±2°
ROLL_RANGE_DEG  = 2     # ±2°
SHIFT_MIN, SHIFT_MAX = 0.002, 0.005   # 平面内平移：模型最大尺寸的 0.2%~0.5%

# 材质微扰（可固定也可轻微变化）
DIFFUSE_BASE = 1.0
AMBIENT_BASE = 0.1
DIFFUSE_JITTER = 0.05   # ±5%
AMBIENT_JITTER = 0.03   # ±3%

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def vec_normalize(v):
    n = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]) + 1e-12
    return [v[0]/n, v[1]/n, v[2]/n]

def vec_cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def projected_max_span(bounds, axis='y'):
    dx = bounds[1]-bounds[0]
    dy = bounds[3]-bounds[2]
    dz = bounds[5]-bounds[4]
    if axis == 'y':   # 看向 +Y/-Y，图像平面是 X-Z
        return max(dx, dz)
    elif axis == 'x': # 看向 +X/-X，图像平面是 Y-Z
        return max(dy, dz)
    else:             # 看向 +Z/-Z，图像平面是 X-Y
        return max(dx, dy)

def fit_camera_ortho(center, bounds, axis='y', sign=+1, margin=1.08):
    """ 正交投影下把模型稳稳装入画面（方形窗口） """
    camera.ParallelProjectionOn()

    # 计算视口所需尺度（正交半高）
    span = projected_max_span(bounds, axis=axis)
    camera.SetParallelScale(0.5 * span * margin)

    # 放置相机在指定轴的远处（距离大小不影响缩放）
    dx = bounds[1]-bounds[0]
    dy = bounds[3]-bounds[2]
    dz = bounds[5]-bounds[4]
    diag = math.sqrt(dx*dx + dy*dy + dz*dz)
    dist = 2.0 * diag  # 足够大的距离，避免裁剪面问题

    pos = [center[0], center[1], center[2]]
    if axis == 'y':
        pos[1] += sign * dist
        camera.SetViewUp(0, 0, 1)
    elif axis == 'x':
        pos[0] += sign * dist
        camera.SetViewUp(0, 0, 1)
    else:  # 'z'
        pos[2] += sign * dist
        camera.SetViewUp(0, 1, 0)

    camera.SetFocalPoint(center)
    camera.SetPosition(pos)
    renderer.ResetCameraClippingRange()

def apply_camera_perturbations_profile_ortho(max_dim, profile="P1"):
    """
    在正交投影下施加小幅差异化扰动（保证不出框）：
    - 角度：取小范围边界组合
    - 平移：取小幅最大/最小并改变方向
    """
    if profile == "P1":
        yaw, pitch, roll = +YAW_RANGE_DEG, 0.0, +ROLL_RANGE_DEG
        shift_r, shift_u = +SHIFT_MAX, +SHIFT_MIN
    elif profile == "P2":
        yaw, pitch, roll = -YAW_RANGE_DEG, +PITCH_RANGE_DEG, -ROLL_RANGE_DEG
        shift_r, shift_u = -SHIFT_MAX, -SHIFT_MIN
    elif profile == "P3":
        yaw, pitch, roll = 0.0, +PITCH_RANGE_DEG, -ROLL_RANGE_DEG
        shift_r, shift_u = +SHIFT_MIN, -SHIFT_MAX
    else:  # P4
        yaw, pitch, roll = 0.0, -PITCH_RANGE_DEG, +ROLL_RANGE_DEG
        shift_r, shift_u = -SHIFT_MIN, +SHIFT_MAX

    # 角度扰动（绕焦点）
    camera.Yaw(yaw)
    camera.Pitch(pitch)
    camera.Roll(roll)

    # 平面内平移（Right/Up 方向），同时移动 position 和 focal
    dop = vec_normalize(list(camera.GetDirectionOfProjection()))
    up  = vec_normalize(list(camera.GetViewUp()))
    right = vec_normalize(vec_cross(dop, up))
    shift = [
        (right[0]*shift_r + up[0]*shift_u) * max_dim,
        (right[1]*shift_r + up[1]*shift_u) * max_dim,
        (right[2]*shift_r + up[2]*shift_u) * max_dim,
    ]
    pos = list(camera.GetPosition()); foc = list(camera.GetFocalPoint())
    camera.SetPosition(pos[0]+shift[0], pos[1]+shift[1], pos[2]+shift[2])
    camera.SetFocalPoint(foc[0]+shift[0], foc[1]+shift[1], foc[2]+shift[2])

def process_stl(stl_path, out_prefix):
    # 清空场景
    renderer.RemoveAllViewProps()

    # 读取 STL
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    # 加法线（可选，视觉更好）
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(reader.GetOutputPort())
    normals.SplittingOff(); normals.ConsistencyOn(); normals.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(normals.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)

    # 模型范围与中心
    bounds = actor.GetBounds()
    center = [(bounds[1] + bounds[0]) / 2,
              (bounds[3] + bounds[2]) / 2,
              (bounds[5] + bounds[4]) / 2]
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])

    # 材质（微扰）
    prop = actor.GetProperty()
    diffuse = clamp(DIFFUSE_BASE * (1.0 + (2*DIFFUSE_JITTER*(0.5)-DIFFUSE_JITTER)), 0.0, 1.0)
    ambient = clamp(AMBIENT_BASE * (1.0 + (2*AMBIENT_JITTER*(0.5)-AMBIENT_JITTER)), 0.0, 1.0)
    prop.SetDiffuse(diffuse)
    prop.SetAmbient(ambient)
    prop.SetSpecular(0.15)
    prop.SetSpecularPower(20)

    # —— 在同一基准视角（+Y）下保存 4 次，使用不同扰动配置 ——
    profiles = ["P1", "P2", "P3", "P4"]

    for idx, prof in enumerate(profiles, start=1):
        # 每次从“基准”重新开始，避免扰动累积
        fit_camera_ortho(center, bounds, axis='y', sign=+1, margin=1.08)
        apply_camera_perturbations_profile_ortho(max_dim, profile=prof)
        renderer.ResetCameraClippingRange()

        # 渲染与保存
        render_window.Render()
        w2i.Modified(); w2i.Update()
        out_path = os.path.join(output_dir, f"{out_prefix}_A{idx}.png")
        png_writer.SetFileName(out_path)
        png_writer.SetInputConnection(w2i.GetOutputPort())
        png_writer.Write()

# ==== 主循环 ====
print(f"共 {len(stl_files)} 个 STL，输出目录：{output_dir}")
for fname in tqdm(stl_files, desc="批量渲染", unit="文件"):
    stl_path = os.path.join(input_dir, fname)
    out_prefix = os.path.splitext(fname)[0]
    try:
        process_stl(stl_path, out_prefix)
    except Exception as e:
        err_log = os.path.join(output_dir, "render_errors.log")
        with open(err_log, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] {fname}: {repr(e)}\n")

print("所有 STL 已处理完成 ✅（正交投影，同一基准 +Y 连拍4张，差异化扰动）")
