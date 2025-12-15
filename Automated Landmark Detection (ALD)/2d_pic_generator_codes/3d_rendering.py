import vtk
import os
from tqdm import tqdm

# ==== 输入输出路径 ====
input_dir = "D:/Codes/Skull_Landmarks_TL/skull_models"   # STL 文件所在文件夹
output_dir = "D:/Codes/Skull_Landmarks_TL/skull_models_2d_pics"  # 截图保存的文件夹
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
render_window.SetSize(800, 800)
render_window.SetMultiSamples(8)
render_window.SetOffScreenRendering(1)  # 离屏渲染

camera = renderer.GetActiveCamera()
w2i = vtk.vtkWindowToImageFilter()
w2i.SetInput(render_window)
png_writer = vtk.vtkPNGWriter()

def process_stl(stl_path, out_prefix):
    # 清空场景
    renderer.RemoveAllViewProps()

    # 读取 STL
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)

    # 相机设置
    bounds = actor.GetBounds()
    center = [(bounds[1] + bounds[0]) / 2,
              (bounds[3] + bounds[2]) / 2,
              (bounds[5] + bounds[4]) / 2]
    camera.SetFocalPoint(center)
    camera.SetViewUp(0, 0, 1)
    renderer.ResetCameraClippingRange()

    def save_view(y_offset, suffix):
        camera.SetPosition(center[0], center[1] + y_offset, center[2])
        renderer.ResetCameraClippingRange()
        render_window.Render()
        w2i.Modified()
        w2i.Update()
        out_path = os.path.join(output_dir, f"{out_prefix}{suffix}.png")
        png_writer.SetFileName(out_path)
        png_writer.SetInputConnection(w2i.GetOutputPort())
        png_writer.Write()

    # 保存两张图
    save_view(+500, "_A")
    save_view(-500, "_B")

# ==== 主循环（带进度条） ====
print(f"共 {len(stl_files)} 个 STL，输出目录：{output_dir}")
for fname in tqdm(stl_files, desc="批量渲染", unit="文件"):
    stl_path = os.path.join(input_dir, fname)
    out_prefix = os.path.splitext(fname)[0]
    try:
        process_stl(stl_path, out_prefix)
    except Exception as e:
        # 不中断批处理，记录错误
        err_log = os.path.join(output_dir, "render_errors.log")
        with open(err_log, "a", encoding="utf-8") as f:
            f.write(f"[ERROR] {fname}: {repr(e)}\n")

print("所有 STL 已处理完成 ✅（离屏渲染 + 进度条）")