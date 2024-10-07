import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class App:
    def __init__(self):
        gui.Application.instance.initialize()

        self.window = gui.Application.instance.create_window("Calibration", 800, 600)

        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)

        em = self.window.theme.font_size

        self._pannel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em , 0.25 * em))
        for i in range(10):
            self._pannel.add_child(gui.Label(f"{i}-th label"))

        self._left_side = gui.Vert()

        self._button = gui.Button("button")
        self._left_side.add_child(self._pannel)
        self._left_side.add_child(self._button)

        self.main = gui.Horiz()
        self.main.add_child(self._left_side)
        self.main.add_fixed(0.25 * em)
        self.main.add_child(self.scene)
        self.window.add_child(self.main)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
        material = rendering.MaterialRecord()

        self.scene.scene.add_geometry("mesh", mesh, material)

        bounds = mesh.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bounds, bounds.get_center())

    def run(self):
        gui.Application.instance.run()


if __name__ == '__main__':
    app = App()
    app.run()
