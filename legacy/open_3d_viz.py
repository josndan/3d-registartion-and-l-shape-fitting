import open3d as o3d
import numpy as np
import time

class Visualizer:
    # palatte = [[1, 0.706, 0],[0, 0.651, 0.929]]
    default_color = [1, 0.706, 0]

    def __init__(self, rotation_animation=False, unique_color=True):
        self.rotation_animation = rotation_animation
        self.unique_color = unique_color
        self.num_geo = 0


    def __enter__(self):
        
        self.viewer = o3d.visualization.Visualizer()
        self.viewer.create_window()
        self.add_axis()
            
        self.opt = self.viewer.get_render_option()
        self.opt.point_size = 2
        self.opt.background_color = np.asarray([0.5, 0.5, 0.5])

        self.ctrl = self.viewer.get_view_control()

        return self

    def add_axis(self):
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=10, origin=[0, 0, 0])
        self.viewer.add_geometry(axis)
    
    def add_pcd(self, pcd, color=None):
        if self.unique_color:
            # pcd.paint_uniform_color(np.random.rand(3))
            pcd.paint_uniform_color(Visualizer.palatte[self.num_geo])
        elif color is not None:
            pcd.paint_uniform_color(color)
        else:
            pcd.paint_uniform_color(Visualizer.default_color)
        self.viewer.add_geometry(pcd)

        self.num_geo +=1

    def run(self):
        if self.rotation_animation:
            while self.viewer.poll_events():
                self.ctrl.rotate(5, 0)
                self.viewer.update_renderer()
        else:
            self.viewer.run()
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.viewer.destroy_window()
    
    palatte = np.array([[1.0, 0.0, 0.0],    # Red
                  [0.0, 1.0, 0.0],    # Green
                  [0.0, 0.0, 1.0],    # Blue
                  [0.5, 0.0, 0.0],    # Maroon
                  [0.0, 0.5, 0.0],    # Forest Green
                  [0.0, 0.0, 0.5],    # Navy Blue
                  [0.5, 0.5, 0.5],    # Gray
                  [1.0, 1.0, 0.0],    # Yellow
                  [1.0, 0.5, 0.0],    # Orange
                  [0.5, 1.0, 0.0],    # Lime
                  [0.5, 0.5, 1.0],    # Light Blue
                  [1.0, 0.0, 0.5],    # Purple
                  [0.0, 1.0, 0.5],    # Turquoise
                  [1.0, 1.0, 1.0],    # White
                  [0.0, 0.0, 0.0],    # Black
                  [1.0, 0.27, 0.0],   # Deep Orange
                  [0.2, 0.6, 0.8],    # Light Sky Blue
                  [1.0, 0.72, 0.77],  # Pink
                  [0.6, 0.6, 0.6],    # Gray
                  [0.2, 0.2, 0.2],    # Dark Gray
                  [0.9, 0.9, 0.9],    # Light Gray
                  [0.6, 0.8, 1.0],    # Light Steel Blue
                  [1.0, 0.84, 0.0],   # Golden
                  [0.0, 0.65, 0.31],  # Dark Olive Green
                  [0.66, 0.66, 0.66], # Silver
                  [0.74, 0.83, 0.9],  # Light Grey
                  [0.78, 1.0, 0.86],  # Pale Green
                  [1.0, 0.0, 1.0],    # Magenta
                  [0.0, 1.0, 1.0],    # Cyan
                  [0.89, 0.47, 0.76], # Lavender
                  [1.0, 0.75, 0.8],   # Peach
                  [0.76, 0.87, 0.78], # Light Olive
                  [0.93, 0.93, 0.93], # Gainsboro
                  [0.8, 0.8, 0.8],    # Dark Gray
                  [1.0, 0.94, 0.0],   # Yellow
                  [0.0, 0.98, 0.6],   # Lime Green
                  [0.49, 0.48, 0.47], # Dark Gray
                  [0.9, 0.0, 1.0],    # Lavender
                  [0.0, 0.9, 0.9],    # Mint Green
                  [0.8, 0.4, 0.0],    # Sienna
                  [0.7, 0.0, 0.7],    # Dark Violet
                  [0.9, 0.6, 0.6],    # Light Coral
                  [0.6, 0.2, 0.8],    # Purple
                  [0.7, 0.5, 0.9],    # Lavender Blush
                  [0.8, 0.8, 0.0],    # Olive
                  [0.2, 0.6, 0.2],    # Dark Sea Green
                  [0.9, 0.9, 0.9],    # Gainsboro
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.6, 0.6, 0.6],    # Dim Gray
                  [0.2, 0.2, 0.2],    # Black
                  [0.9, 0.6, 0.0],    # Dark Orange
                  [0.9, 0.9, 0.0],    # Dark Khaki
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.8, 0.8, 0.8],    # Light Gray
                  [0.0, 0.0, 0.9],    # Dark Blue
                  [0.6, 0.6, 0.6],    # Gray
                  [0.9, 0.3, 0.3],    # Indian Red
                  [0.2, 0.8, 0.2],    # Lime Green
                  [0.7, 0.7, 0.7],    # Gray
                  [0.6, 0.2, 0.2],    # Maroon
                  [0.2, 0.2, 0.6],    # Medium Blue
                  [0.7, 0.4, 0.4],    # Rosy Brown
                  [0.9, 0.7, 0.9],    # Lavender
                  [0.3, 0.3, 0.3],    # Dark Gray
                  [0.0, 0.0, 0.6],    # Navy
                  [0.8, 0.2, 0.2],    # Brown
                  [0.4, 0.4, 0.4],    # Dark Gray
                  [0.6, 0.6, 0.6],    # Gray
                  [0.2, 0.2, 0.2],    # Black
                  [0.9, 0.9, 0.0],    # Yellow
                  [0.7, 0.7, 0.7],    # Gray
                  [0.5, 0.5, 0.5],    # Gray
                  [0.6, 0.6, 0.6],    # Gray
                  [0.0, 0.5, 0.5],    # Teal
                  [0.9, 0.9, 0.9],    # White Smoke
                  [0.6, 0.2, 0.8],    # Purple
                  [0.7, 0.5, 0.9],    # Lavender Blush
                  [0.8, 0.8, 0.0],])  # Olive

    @staticmethod
    def visualize_pcds(pcds,unique_color):    
        with Visualizer(unique_color=unique_color) as viz:
            for pcd in pcds:
                viz.add_pcd(pcd)

            viz.run()

    @staticmethod
    def play_pcds(pcds,fps=20):    
        with Visualizer(unique_color=False) as viz:
            
            last_pcd = None
            i = 0
            start = 0

            zoom=0.3412
            front=[0.4257, -0.2125, -0.8795]
            lookat=[2.6172, 2.0475, 1.532]
            up=[-0.0694, -0.9768, 0.2024]

            while viz.viewer.poll_events():
                end = time.time()
                
                if end - start < 1/fps:
                    continue

                if i < len(pcds):
                    pcd = pcds[i]
                    i += 1

                # fov = viz.ctrl.get_field_of_view()
                
                if last_pcd:
                    viz.viewer.remove_geometry(last_pcd)
                

                viz.viewer.add_geometry(pcd)
                # viz.ctrl.change_field_of_view(fov)
                # viz.ctrl.set_zoom(zoom)
                # viz.ctrl.set_lookat(lookat)
                # viz.ctrl.set_front(front)
                # viz.ctrl.set_up(up)

                viz.viewer.update_renderer()
                last_pcd = pcd

                start = time.time()

